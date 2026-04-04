from __future__ import annotations

import json
import os
import re
import ast
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Optional
from urllib import error, parse, request

from .schemas import RealtimeFlightCard

FLIGHT_NO_RE = re.compile(r"(?i)(?<![A-Za-z0-9])([A-Z]{2}\s?\d{3,4})(?![A-Za-z0-9])")
REALTIME_KEYWORDS = (
    "实时",
    "动态",
    "延误",
    "起飞",
    "降落",
    "到达",
    "出发",
    "登机口",
    "航班状态",
    "realtime",
    "live",
    "flight status",
    "delay",
    "arrival",
    "departure",
)


class VariFlightMCPClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("VARIFLIGHT_MCP_URL", "https://ai.variflight.com/servers/aviation/mcp/").strip()
        self.api_key = os.getenv("VARIFLIGHT_MCP_API_KEY", "").strip()
        self.timeout_seconds = float(os.getenv("VARIFLIGHT_MCP_TIMEOUT", "10"))
        self._request_id = 1

    def _url(self) -> str:
        url = self.base_url.strip()
        if "?" in url:
            left, right = url.split("?", 1)
            url = left.rstrip("/") + "?" + right
        else:
            url = url.rstrip("/")
        if self.api_key and "api_key=" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}api_key={parse.quote(self.api_key)}"
        return url

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
        body = json.loads(text or "{}")
        if "error" in body:
            raise RuntimeError(str(body.get("error")))
        return body.get("result", {}) if isinstance(body, dict) else {}

    def _rpc(self, method: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {},
        }
        self._request_id += 1
        url = self._url()
        try:
            return self._post_json(url, payload)
        except error.HTTPError as exc:
            if exc.code not in {307, 308}:
                raise
            location = exc.headers.get("Location")
            if not location:
                raise
            if self.api_key and "api_key=" not in location:
                sep = "&" if "?" in location else "?"
                location = f"{location}{sep}api_key={parse.quote(self.api_key)}"
            return self._post_json(location, payload)

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._rpc("tools/list", {})
        tools = result.get("tools", []) if isinstance(result, dict) else []
        return [t for t in tools if isinstance(t, dict)]

    def select_flight_tool(self) -> Optional[str]:
        tools = self.list_tools()
        priorities = ("flight", "status", "realtime", "dynamic", "航班", "动态")
        for t in tools:
            name = str(t.get("name", ""))
            lname = name.lower()
            if any(p in lname for p in priorities) or any(p in name for p in ("航班", "状态", "实时")):
                return name
        if tools:
            return str(tools[0].get("name", "")) or None
        return None

    def has_tool(self, name: str) -> bool:
        return any(str(t.get("name", "")) == name for t in self.list_tools())

    def get_today_date(self) -> str:
        if self.has_tool("getTodayDate"):
            try:
                result = self.call_tool("getTodayDate", {"random_string": "today"})
                txt = _extract_display_text(result)
                m = re.search(r"\d{4}-\d{2}-\d{2}", txt)
                if m:
                    return m.group(0)
                date_val = _find_value(result, ("date", "today", "current_date"))
                if date_val:
                    m2 = re.search(r"\d{4}-\d{2}-\d{2}", str(date_val))
                    if m2:
                        return m2.group(0)
            except Exception:
                pass
        return datetime.now().strftime("%Y-%m-%d")

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return self._rpc("tools/call", {"name": name, "arguments": arguments})


def normalize_flight_no(text: str) -> Optional[str]:
    m = FLIGHT_NO_RE.search(text or "")
    if not m:
        return None
    return m.group(1).replace(" ", "").upper()


def is_realtime_flight_question(question: str) -> bool:
    q = (question or "").lower()
    return any(k in q for k in REALTIME_KEYWORDS) or bool(normalize_flight_no(question))


def _iter_dicts(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _iter_dicts(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_dicts(item)


def _find_value(result: Any, keys: tuple[str, ...]) -> Any:
    keyset = {k.lower() for k in keys}
    for d in _iter_dicts(result):
        for k, v in d.items():
            if str(k).lower() in keyset and v not in (None, ""):
                return v
    return None


def _to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    s = str(value)
    m = re.search(r"-?\d+", s)
    if not m:
        return None
    try:
        return int(m.group(0))
    except ValueError:
        return None


def _to_text(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    return str(value).strip()


def _parse_dt(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _extract_display_text(result: dict[str, Any]) -> str:
    content = result.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for c in content:
            if isinstance(c, dict):
                if c.get("type") == "text" and c.get("text"):
                    parts.append(str(c.get("text")))
                elif c.get("text"):
                    parts.append(str(c.get("text")))
            elif c:
                parts.append(str(c))
        if parts:
            return "\n".join(parts).strip()
    for key in ("text", "answer", "result", "message"):
        val = result.get(key)
        if val:
            return str(val).strip()
    return ""


def _parse_embedded_dict_from_text(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        parsed = ast.literal_eval(candidate)
    except Exception:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _card_is_sparse(card: RealtimeFlightCard) -> bool:
    fields = [
        card.status,
        card.planned_departure,
        card.actual_departure,
        card.planned_arrival,
        card.actual_arrival,
        card.terminal,
        card.gate,
    ]
    return all(v in (None, "") for v in fields) and card.delay_minutes is None


def _looks_like_tool_error(text: str) -> bool:
    t = (text or "").lower()
    return (
        "error executing tool" in t
        or "validation error" in t
        or "field required" in t
        or "not acceptable" in t
    )


def _build_flight_card(result: dict[str, Any], fallback_flight_no: Optional[str]) -> RealtimeFlightCard:
    flight_no = _to_text(
        _find_value(result, ("flight_no", "flightNo", "flight", "flight_num", "flightNumber", "FlightNo"))
    ) or fallback_flight_no or "UNKNOWN"

    planned_departure = _to_text(
        _find_value(result, ("scheduled_departure", "std", "planned_departure", "plan_departure_time", "dep_plan_time", "FlightDeptimePlanDate"))
    )
    actual_departure = _to_text(
        _find_value(result, ("actual_departure", "atd", "real_departure_time", "dep_actual_time", "FlightDeptimeDate"))
    )
    planned_arrival = _to_text(
        _find_value(result, ("scheduled_arrival", "sta", "planned_arrival", "plan_arrival_time", "arr_plan_time", "FlightArrtimePlanDate"))
    )
    actual_arrival = _to_text(
        _find_value(result, ("actual_arrival", "ata", "real_arrival_time", "arr_actual_time", "FlightArrtimeDate"))
    )

    delay_minutes = _to_int(
        _find_value(result, ("delay_minutes", "delay", "delay_min", "delayTime", "delay_time", "DelayTime"))
    )
    if delay_minutes is None:
        dep_actual_dt = _parse_dt(actual_departure)
        dep_plan_dt = _parse_dt(planned_departure)
        arr_actual_dt = _parse_dt(actual_arrival)
        arr_plan_dt = _parse_dt(planned_arrival)
        if dep_actual_dt and dep_plan_dt:
            delay_minutes = int((dep_actual_dt - dep_plan_dt).total_seconds() // 60)
        elif arr_actual_dt and arr_plan_dt:
            delay_minutes = int((arr_actual_dt - arr_plan_dt).total_seconds() // 60)

    status = _to_text(_find_value(result, ("status", "flight_status", "state", "status_text", "FlightState", "AssistFlightState")))

    dep_terminal = _to_text(_find_value(result, ("departure_terminal", "dep_terminal", "terminal_dep", "depTerminal", "FlightHTerminal")))
    arr_terminal = _to_text(_find_value(result, ("arrival_terminal", "arr_terminal", "terminal_arr", "arrTerminal", "FlightTerminal")))
    terminal = None
    if dep_terminal and arr_terminal:
        terminal = f"出发 {dep_terminal} / 到达 {arr_terminal}"
    else:
        terminal = dep_terminal or arr_terminal

    dep_gate = _to_text(_find_value(result, ("departure_gate", "dep_gate", "gate", "boarding_gate", "gate_dep", "BoardGate")))
    arr_gate = _to_text(_find_value(result, ("arrival_gate", "arr_gate", "gate_arr", "ReachExit")))
    gate = None
    if dep_gate and arr_gate:
        gate = f"出发 {dep_gate} / 到达 {arr_gate}"
    else:
        gate = dep_gate or arr_gate

    return RealtimeFlightCard(
        flight_no=flight_no,
        status=status,
        planned_departure=planned_departure,
        actual_departure=actual_departure,
        planned_arrival=planned_arrival,
        actual_arrival=actual_arrival,
        delay_minutes=delay_minutes,
        terminal=terminal,
        gate=gate,
    )


def query_realtime_flight(question: str, flight_no: Optional[str] = None) -> tuple[str, RealtimeFlightCard] | None:
    if not is_realtime_flight_question(question):
        return None

    guessed_flight_no = flight_no or normalize_flight_no(question)
    client = VariFlightMCPClient()
    if guessed_flight_no and client.has_tool("searchFlightsByNumber"):
        tool_name = "searchFlightsByNumber"
    else:
        tool_name = client.select_flight_tool()
    if not tool_name:
        raise RuntimeError("MCP 未返回可用工具")

    if tool_name == "searchFlightsByNumber":
        date_str = client.get_today_date()
        arg_candidates = [
            {"fnum": guessed_flight_no, "date": date_str},
            {"fnum": guessed_flight_no, "date": date_str, "dep": "", "arr": ""},
        ]
    elif tool_name == "searchFlightsByDepArr":
        date_str = client.get_today_date()
        arg_candidates = [
            {"date": date_str, "depcity": "广州", "arrcity": "北京"},
            {"date": date_str, "question": question},
            {"date": date_str, "query": question},
        ]
    else:
        arg_candidates = [
            {"question": question, "flight_no": guessed_flight_no},
            {"query": question, "flight_no": guessed_flight_no},
            {"flight_no": guessed_flight_no},
            {"question": question},
            {"query": question},
        ]

    last_error: Optional[Exception] = None
    result: Optional[dict[str, Any]] = None
    for args in arg_candidates:
        filtered = {k: v for k, v in args.items() if v not in (None, "")}
        try:
            result = client.call_tool(tool_name, filtered)
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue

    if result is None:
        raise RuntimeError(f"实时航班查询失败: {last_error}")

    text = _extract_display_text(result)
    card = _build_flight_card(result, guessed_flight_no)
    if _card_is_sparse(card):
        embedded = _parse_embedded_dict_from_text(text)
        if embedded:
            card = _build_flight_card(embedded, guessed_flight_no)

    if _looks_like_tool_error(text):
        raise RuntimeError(text)

    if not text:
        lines = [
            f"航班号：{card.flight_no}",
            f"计划起飞：{card.planned_departure or '未知'}",
            f"实际起飞：{card.actual_departure or '未知'}",
            f"计划到达：{card.planned_arrival or '未知'}",
            f"实际到达：{card.actual_arrival or '未知'}",
            f"延误分钟：{card.delay_minutes if card.delay_minutes is not None else '未知'}",
            f"航站楼：{card.terminal or '未知'}",
            f"登机口：{card.gate or '未知'}",
        ]
        text = "\n".join(lines)

    return text, card
