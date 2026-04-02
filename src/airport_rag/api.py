from __future__ import annotations

import re
import mimetypes
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from typing import List
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .schemas import AskRequest, AskResponse, HealthResponse, IngestRequest, IngestResponse
from .service import AirportRAGService


app = FastAPI(title="Airport KB RAG Assistant", version="1.0.0")
service = AirportRAGService()
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DOC_ROOT = (BASE_DIR.parent.parent / "data" / "documents").resolve()

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


SEED_SELF_TEST_CASES = [
    # 海关（6）
    {"topic": "海关", "question": "入境最多能携带多少现金？", "expect": "answer"},
    {"topic": "海关", "question": "人民币和外币现金分别超过多少需要申报？", "expect": "answer"},
    {"topic": "海关", "question": "红色通道和绿色通道怎么选择？", "expect": "answer"},
    {"topic": "海关", "question": "海关申报单在什么情况下必须填写？", "expect": "answer"},
    {"topic": "海关", "question": "海关罚款标准是多少钱？", "expect": "low-confidence"},
    {"topic": "海关", "question": "海关窗口晚上几点下班？", "expect": "low-confidence"},

    # 边防（5）
    {"topic": "边防", "question": "港澳居民来往内地应该持什么证件？", "expect": "answer"},
    {"topic": "边防", "question": "外国人入境是否需要填写入境卡？", "expect": "answer"},
    {"topic": "边防", "question": "外国籍港澳居民来往内地能停留多久？", "expect": "answer"},
    {"topic": "边防", "question": "外国人入境卡在哪里领取？", "expect": "low-confidence"},
    {"topic": "边防", "question": "边检人工通道平均排队多久？", "expect": "low-confidence"},

    # 出发（4）
    {"topic": "出发", "question": "国际出发建议提前多久到达航站楼？", "expect": "answer"},
    {"topic": "出发", "question": "国内出发建议提前多久到达航站楼？", "expect": "answer"},
    {"topic": "出发", "question": "值机柜台一般什么时候关闭？", "expect": "low-confidence"},
    {"topic": "出发", "question": "机场有吸烟区吗？", "expect": "low-confidence"},

    # 行李（5）
    {"topic": "行李", "question": "充电宝120Wh能带吗？", "expect": "answer"},
    {"topic": "行李", "question": "超过160Wh充电宝能带吗？", "expect": "answer"},
    {"topic": "行李", "question": "打火机可以随身携带吗？", "expect": "answer"},
    {"topic": "行李", "question": "行李超重费用是多少？", "expect": "low-confidence"},
    {"topic": "行李", "question": "托运行李每公斤加收多少钱？", "expect": "low-confidence"},

    # 航司（CZ + 9C，共5）
    {"topic": "航司", "question": "南航客服热线是多少？", "expect": "answer"},
    {"topic": "航司", "question": "南航境外客服电话是多少？", "expect": "answer"},
    {"topic": "航司", "question": "春秋航空是全经济舱吗？", "expect": "answer"},
    {"topic": "航司", "question": "春秋航空是否提供免费餐饮？", "expect": "answer"},
    {"topic": "航司", "question": "春秋航空客服电话是多少？", "expect": "low-confidence"},
]


def _expand_self_test_cases(seed_cases: list[dict]) -> list[dict]:
    variants = [
        lambda q: q,
        lambda q: f"请问{q}",
        lambda q: f"依据现有规则，{q}",
        lambda q: f"{q}（请附依据）",
    ]
    expanded: list[dict] = []
    for case in seed_cases:
        for transform in variants:
            expanded.append(
                {
                    "topic": case["topic"],
                    "question": transform(case["question"]),
                    "expect": case["expect"],
                }
            )
    return expanded


DEFAULT_SELF_TEST_CASES = _expand_self_test_cases(SEED_SELF_TEST_CASES)
SELF_TEST_TOPICS = sorted({case["topic"] for case in DEFAULT_SELF_TEST_CASES})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/app")
def app_home() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI page not found")
    return FileResponse(index_file)


@app.get("/admin")
def admin_home() -> FileResponse:
    admin_file = STATIC_DIR / "admin.html"
    if not admin_file.exists():
        raise HTTPException(status_code=404, detail="Admin UI page not found")
    return FileResponse(admin_file)


class AdminClassifyRequest(BaseModel):
    filename: str = Field(min_length=1)
    content: str = ""
    scope_hint: Optional[str] = None
    carrier_hint: Optional[str] = None


class AdminCreateDocRequest(AdminClassifyRequest):
    auto_sync: bool = True


class AdminUpdateDocRequest(BaseModel):
    content: str
    auto_sync: bool = True


def _normalize_text_key(text: str) -> str:
    return re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text or "").lower()


def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="filename is required")
    if name.startswith("."):
        raise HTTPException(status_code=400, detail="hidden filename is not allowed")
    return name


def _safe_doc_path(relative_path: str) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute():
        raise HTTPException(status_code=400, detail="path must be relative")
    candidate = (DOC_ROOT / rel).resolve()
    if DOC_ROOT not in candidate.parents and candidate != DOC_ROOT:
        raise HTTPException(status_code=400, detail="invalid path")
    return candidate


def _load_carrier_alias_map() -> dict[str, str]:
    aliases = {
        "南航": "CZ",
        "中国南方航空": "CZ",
        "南方航空": "CZ",
        "东航": "MU",
        "国航": "CA",
        "春秋": "9C",
        "春秋航空": "9C",
    }
    code_doc = DOC_ROOT / "airport" / "航司代码"
    if code_doc.exists():
        try:
            text = code_doc.read_text(encoding="utf-8")
            for raw_line in text.splitlines():
                line = raw_line.strip()
                if not line.startswith("|") or line.startswith("|---"):
                    continue
                cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
                if len(cells) < 2:
                    continue

                code_idx = -1
                code_value = ""
                for i, cell in enumerate(cells):
                    candidate = cell.strip().upper()
                    if re.fullmatch(r"[A-Z0-9]{2}", candidate) and candidate != "WH":
                        code_idx = i
                        code_value = candidate
                        break
                if code_idx < 0:
                    continue

                for i, cell in enumerate(cells):
                    if i == code_idx:
                        continue
                    raw = cell.strip()
                    if not raw:
                        continue
                    candidates = {raw, raw.lower()}
                    for suffix in ["股份有限公司", "有限责任公司", "航空公司", "航空", "公司"]:
                        if suffix in raw:
                            candidates.add(raw.replace(suffix, ""))
                    if raw.startswith("中国") and len(raw) > 2:
                        candidates.add(raw[2:])
                    for c in candidates:
                        c_norm = _normalize_text_key(c)
                        if c_norm:
                            aliases[c_norm] = code_value
        except Exception:
            pass
    normalized = {}
    for k, v in aliases.items():
        nk = _normalize_text_key(k)
        if nk:
            normalized[nk] = v.upper()
    return normalized


def _resolve_carrier_code(text: str) -> Optional[str]:
    normalized = _normalize_text_key(text)
    alias_map = _load_carrier_alias_map()
    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        if alias and alias in normalized:
            return alias_map[alias]

    m = re.search(r"(?<![A-Za-z0-9])([A-Za-z0-9]{2})(?![A-Za-z0-9])", text or "")
    if m:
        code = m.group(1).upper()
        if code != "WH":
            return code
    return None


def _classify_target(filename: str, content: str, scope_hint: Optional[str], carrier_hint: Optional[str]) -> tuple[str, str, str]:
    if scope_hint:
        s = scope_hint.strip().lower()
        if s == "airport":
            return "airport", "", "airport"
        if s == "airline":
            carrier = _resolve_carrier_code(carrier_hint or filename or content)
            if not carrier:
                raise HTTPException(status_code=400, detail="scope_hint=airline requires carrier_hint/name/code")
            return "airline", carrier, carrier

    merged = "\n".join([filename or "", content or "", carrier_hint or ""])
    carrier = _resolve_carrier_code(merged)
    if carrier:
        return "airline", carrier, carrier

    airport_keywords = ["白云机场", "航站楼", "海关", "边检", "边防", "机场", "安检"]
    if any(k in merged for k in airport_keywords):
        return "airport", "", "airport"

    return "airport", "", "airport"


def _sync_if_needed(auto_sync: bool) -> dict:
    if not auto_sync:
        return {"synced": False}
    result = service.ingest("./data/documents")
    return {
        "synced": True,
        "indexed_chunks": result.indexed_chunks,
        "processed_files": result.processed_files,
    }


def _is_likely_text_file(path: Path) -> bool:
    text_exts = {
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".json",
        ".yaml",
        ".yml",
        ".xml",
        ".html",
        ".htm",
    }
    return path.suffix.lower() in text_exts or path.suffix == ""


def _guess_media_type(path: Path) -> str:
    media_type, _ = mimetypes.guess_type(str(path))
    return media_type or "application/octet-stream"


@app.get("/admin/docs")
def admin_list_docs() -> dict:
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    docs = []
    for p in sorted([x for x in DOC_ROOT.rglob("*") if x.is_file()]):
        rel = p.relative_to(DOC_ROOT).as_posix()
        if any(part.startswith(".") for part in rel.split("/")):
            continue
        parts = rel.split("/")
        scope = "airport" if parts and parts[0].lower() == "airport" else "airline"
        carrier = "" if scope == "airport" else parts[0].upper()
        stat = p.stat()
        docs.append(
            {
                "path": rel,
                "scope": scope,
                "carrier": carrier,
                "size": stat.st_size,
                "updated_at": int(stat.st_mtime),
            }
        )
    return {"root": str(DOC_ROOT), "total": len(docs), "documents": docs}


def _build_tree_node(path: Path, root: Path) -> dict:
    node = {"name": path.name, "path": path.relative_to(root).as_posix()}
    if path.is_dir():
        children = []
        for p in sorted([x for x in path.iterdir() if not x.name.startswith('.')]):
            children.append(_build_tree_node(p, root))
        node["type"] = "dir"
        node["children"] = children
    else:
        node["type"] = "file"
        node["size"] = path.stat().st_size
    return node


@app.get("/admin/tree")
def admin_tree() -> dict:
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    tree = _build_tree_node(DOC_ROOT, DOC_ROOT)
    return {"root": str(DOC_ROOT), "tree": tree}


@app.get("/admin/search")
def admin_search(q: str = Query(..., min_length=1), limit: int = 50) -> dict:
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    qnorm = q.lower()
    matches = []
    for p in DOC_ROOT.rglob("*"):
        if not p.is_file() or p.name.startswith('.'):
            continue
        rel = p.relative_to(DOC_ROOT).as_posix()
        if qnorm in rel.lower():
            matches.append({"path": rel, "score": 100, "snippet": ""})
            if len(matches) >= limit:
                break
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            try:
                text = p.read_bytes().decode("gb18030")
            except Exception:
                continue
        idx = text.lower().find(qnorm)
        if idx >= 0:
            start = max(0, idx - 40)
            snippet = text[start: start + 160].replace('\n', ' ')
            matches.append({"path": rel, "score": 50, "snippet": snippet})
            if len(matches) >= limit:
                break
    return {"query": q, "total": len(matches), "matches": matches}


@app.post("/admin/docs/bulk")
def admin_bulk_upload(files: List[UploadFile] = File(...), auto_sync: bool = Form(True)) -> dict:
    created = []
    errors = []
    for f in files:
        fname = _sanitize_filename(f.filename)
        try:
            body = f.file.read()
            suffix = Path(fname).suffix.lower()
            is_text = suffix in {"", ".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml", ".xml", ".html", ".htm"}
            content = ""
            if is_text:
                try:
                    content = body.decode("utf-8")
                except UnicodeDecodeError:
                    content = body.decode("gb18030", errors="ignore")
            scope, carrier, folder = _classify_target(fname, content, None, None)
            target = _safe_doc_path(f"{folder}/{fname}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if is_text:
                target.write_text(content, encoding="utf-8")
            else:
                target.write_bytes(body)
            created.append(
                {
                    "path": target.relative_to(DOC_ROOT).as_posix(),
                    "carrier": carrier,
                    "binary": not is_text,
                }
            )
        except Exception as exc:
            errors.append({"file": f.filename, "error": str(exc)})
    sync = _sync_if_needed(auto_sync)
    return {"created": created, "errors": errors, **sync}


@app.get("/admin/docs/content")
def admin_get_doc_content(path: str = Query(..., min_length=1)) -> dict:
    full = _safe_doc_path(path)
    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail="document not found")

    rel = full.relative_to(DOC_ROOT).as_posix()
    media_type = _guess_media_type(full)
    preview_url = f"/admin/docs/raw?path={rel}"

    if _is_likely_text_file(full):
        try:
            content = full.read_text(encoding="utf-8")
            return {
                "path": rel,
                "content": content,
                "is_binary": False,
                "media_type": media_type,
                "editable": True,
                "preview_url": preview_url,
            }
        except UnicodeDecodeError:
            try:
                content = full.read_bytes().decode("gb18030")
                return {
                    "path": rel,
                    "content": content,
                    "is_binary": False,
                    "media_type": media_type,
                    "editable": True,
                    "preview_url": preview_url,
                }
            except Exception:
                pass

    return {
        "path": rel,
        "content": "",
        "is_binary": True,
        "media_type": media_type,
        "editable": False,
        "preview_url": preview_url,
    }


@app.get("/admin/docs/raw")
def admin_get_doc_raw(path: str = Query(..., min_length=1)) -> FileResponse:
    full = _safe_doc_path(path)
    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail="document not found")
    return FileResponse(path=full, media_type=_guess_media_type(full), filename=full.name)


@app.post("/admin/docs/classify")
def admin_classify_doc(req: AdminClassifyRequest) -> dict:
    filename = _sanitize_filename(req.filename)
    scope, carrier, folder = _classify_target(filename, req.content, req.scope_hint, req.carrier_hint)
    return {
        "scope": scope,
        "carrier": carrier,
        "target_folder": folder,
        "target_path": f"{folder}/{filename}",
    }


@app.post("/admin/docs")
def admin_create_doc(req: AdminCreateDocRequest) -> dict:
    filename = _sanitize_filename(req.filename)
    scope, carrier, folder = _classify_target(filename, req.content, req.scope_hint, req.carrier_hint)
    target = _safe_doc_path(f"{folder}/{filename}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(req.content or "", encoding="utf-8")
    sync_result = _sync_if_needed(req.auto_sync)
    return {
        "path": target.relative_to(DOC_ROOT).as_posix(),
        "scope": scope,
        "carrier": carrier,
        **sync_result,
    }


@app.put("/admin/docs/content")
def admin_update_doc(path: str = Query(..., min_length=1), req: AdminUpdateDocRequest = None) -> dict:
    if req is None:
        raise HTTPException(status_code=400, detail="request body required")
    full = _safe_doc_path(path)
    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail="document not found")
    full.write_text(req.content or "", encoding="utf-8")
    sync_result = _sync_if_needed(req.auto_sync)
    return {"path": full.relative_to(DOC_ROOT).as_posix(), **sync_result}


@app.delete("/admin/docs")
def admin_delete_doc(path: str = Query(..., min_length=1), auto_sync: bool = True) -> dict:
    full = _safe_doc_path(path)
    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail="document not found")
    full.unlink()
    sync_result = _sync_if_needed(auto_sync)
    return {"path": path, **sync_result}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    try:
        return service.ingest(req.input_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ingest/default", response_model=IngestResponse)
def ingest_default() -> IngestResponse:
    try:
        return service.ingest("./data/documents")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        return service.ask(req.question, req.top_k)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/self-test")
def self_test() -> dict:
    results = []
    passed = 0
    topic_scores = {
        topic: {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate": 0.0,
        }
        for topic in SELF_TEST_TOPICS
    }

    for case in DEFAULT_SELF_TEST_CASES:
        topic = case.get("topic", "未分类")
        q = case["question"]
        expect = case["expect"]
        if topic in topic_scores:
            topic_scores[topic]["total"] += 1
        try:
            r = service.ask(q)
            is_low = r.confidence_note == "low-confidence"
            actual = "low-confidence" if is_low else "answer"
            ok = actual == expect
            if ok:
                passed += 1
                if topic in topic_scores:
                    topic_scores[topic]["passed"] += 1
            else:
                if topic in topic_scores:
                    topic_scores[topic]["failed"] += 1
            results.append(
                {
                    "topic": topic,
                    "question": q,
                    "expect": expect,
                    "actual": actual,
                    "pass": ok,
                    "confidence": r.confidence_note,
                    "citations": len(r.citations),
                    "first_line": r.answer.split("\n")[0] if r.answer else "",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "topic": topic,
                    "question": q,
                    "expect": expect,
                    "actual": "error",
                    "pass": False,
                    "confidence": "error",
                    "citations": 0,
                    "first_line": str(exc),
                }
            )
            if topic in topic_scores:
                topic_scores[topic]["failed"] += 1

    for topic in SELF_TEST_TOPICS:
        total = topic_scores[topic]["total"]
        passed_topic = topic_scores[topic]["passed"]
        topic_scores[topic]["pass_rate"] = round(passed_topic / total, 4) if total else 0.0

    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "errors": sum(1 for x in results if x["confidence"] == "error"),
        "low_confidence": sum(1 for x in results if x["confidence"] == "low-confidence"),
        "topic_scores": topic_scores,
        "results": results,
    }
