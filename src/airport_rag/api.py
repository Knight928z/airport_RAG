from __future__ import annotations

import json
import hashlib
import re
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from typing import List
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .schemas import (
    AnswerFeedbackRequest,
    AnswerFeedbackResponse,
    AskRequest,
    AskResponse,
    FlightRealtimeRequest,
    HealthResponse,
    IngestRequest,
    IngestResponse,
)
from .service import AirportRAGService
from .ingest import extract_text_from_image
from .realtime_flight import query_realtime_flight


app = FastAPI(title="Airport KB RAG Assistant", version="1.0.0")
service = AirportRAGService()
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = (BASE_DIR.parent.parent / "data").resolve()
STATIC_DIR = BASE_DIR / "static"
DOC_ROOT = (DATA_ROOT / "documents").resolve()
PATCH_ROOT = (DATA_ROOT / "patches").resolve()
FEEDBACK_ROOT = (DATA_ROOT / "feedback").resolve()
ANSWER_FEEDBACK_LOG = FEEDBACK_ROOT / "answer_feedback.jsonl"
UNCOVERED_LOG = FEEDBACK_ROOT / "uncovered_questions.jsonl"
PATCH_DOC_NAME = "用户纠错补丁.md"
PATCH_REGISTRY_LOG = FEEDBACK_ROOT / "patch_registry.jsonl"
PATCH_AUDIT_LOG = FEEDBACK_ROOT / "patch_audit.jsonl"
PATCH_MAX_BYTES = 64 * 1024
PATCH_MERGE_ENTRY_THRESHOLD = 8
_LEGACY_PATCH_MIGRATED = False

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(content)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _normalize_patch_key(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower())


def _feedback_topic(question: str, corrected_answer: str, comment: str) -> str:
    joined = f"{question}\n{corrected_answer}\n{comment}".lower()
    topic_rules = {
        "行李": ["行李", "托运", "随身", "充电宝", "锂电池", "液态", "液体", "公斤"],
        "票务": ["退票", "改签", "保险", "机票", "军残", "残疾军人证", "儿童票", "婴儿票"],
        "海关": ["海关", "申报", "红色通道", "绿色通道", "现钞", "人民币", "美元"],
        "边检": ["边检", "边防", "入境", "出境", "旅游团", "签证", "护照", "入境卡"],
        "出发到达": ["出发", "到达", "值机", "登机", "航站楼", "截止", "提前"],
        "客服": ["客服", "热线", "电话", "联系方式"],
    }
    for topic, keywords in topic_rules.items():
        if any(k in joined for k in keywords):
            return topic
    return "综合"


def _feedback_confidence_tier(confidence_note: str, rating: int, corrected_answer: str) -> str:
    if corrected_answer.strip():
        return "high"
    if rating < 0 and confidence_note in {"low-confidence", "index-empty"}:
        return "high"
    if rating < 0:
        return "medium"
    return "low"


def _feedback_fingerprint(question: str, corrected_answer: str, topic: str) -> str:
    base = "|".join([_normalize_patch_key(question), _normalize_patch_key(corrected_answer), topic])
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _existing_patch_fingerprints() -> set[str]:
    if not PATCH_REGISTRY_LOG.exists():
        return set()
    fingerprints: set[str] = set()
    try:
        for line in PATCH_REGISTRY_LOG.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            fp = row.get("fingerprint")
            if fp:
                fingerprints.add(fp)
    except Exception:
        return set()
    return fingerprints


def _feedback_patch_target(question: str, corrected_answer: str) -> tuple[str, Path]:
    carrier = _resolve_carrier_code(f"{question}\n{corrected_answer}")
    folder = carrier if carrier else "airport"
    topic = _feedback_topic(question, corrected_answer, "")
    tier = _feedback_confidence_tier("", -1, corrected_answer)
    month_key = datetime.now().strftime("%Y-%m")
    rel = f"{folder}/{topic}/{tier}/{month_key}-{PATCH_DOC_NAME}"
    return rel, _safe_patch_path(rel)


def _build_feedback_patch_markdown(
    *,
    question: str,
    answer: str,
    confidence_note: str,
    rating: int,
    corrected_answer: str,
    comment: str,
) -> str:
    ts = datetime.now().isoformat(timespec="seconds")
    quality_signal = "dislike" if rating < 0 else "like"
    correction = corrected_answer if corrected_answer else "（用户未提供具体改写，需基于评价继续优化）"
    angle = comment if comment else "（未提供）"
    return (
        "\n\n---\n"
        f"## feedback-patch {ts}\n"
        f"- 问题：{question}\n"
        f"- 旧答案：{answer}\n"
        f"- 置信标记：{confidence_note}\n"
        f"- 质量信号：{quality_signal}\n"
        f"- 置信分层：{_feedback_confidence_tier(confidence_note, rating, corrected_answer)}\n"
        f"- 评价角度：{angle}\n"
        f"- 建议修正：{correction}\n"
        "- 应用策略：将本条作为知识补丁参与后续检索与回答重写，优先采用“建议修正”中的可核验事实。\n"
    )


def _ensure_patch_header(path: Path, *, topic: str, tier: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# 用户纠错知识补丁\n\n"
        "> 自动由 /feedback 触发生成，用于将用户点踩与纠错内容纳入知识库增量修正。\n"
        f"> 主题：{topic} ｜ 置信分层：{tier}\n",
        encoding="utf-8",
    )


def _count_patch_entries(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return 0
    return text.count("## feedback-patch ")


def _merge_patch_into_main_doc(rel_path: str, patch_file: Path, *, recreate_header: bool = True) -> bool:
    if not patch_file.exists():
        return False
    try:
        text = patch_file.read_text(encoding="utf-8")
    except Exception:
        return False

    entries = [seg for seg in text.split("\n\n---\n") if "## feedback-patch" in seg]
    if not entries:
        return False

    rel_parts = Path(rel_path).parts
    root_folder = rel_parts[0] if rel_parts else "airport"
    main_doc = _safe_doc_path(f"{root_folder}/知识补丁合并稿.md")
    main_doc.parent.mkdir(parents=True, exist_ok=True)
    if not main_doc.exists():
        main_doc.write_text(
            "# 知识补丁合并稿\n\n> 由反馈补丁自动聚合生成，用于人工审核后回写主知识文档。\n",
            encoding="utf-8",
        )

    merged_lines = [
        "\n\n---",
        f"## patch-merge {datetime.now().isoformat(timespec='seconds')}",
        f"- 来源补丁：{rel_path}",
    ]
    for e in entries[-20:]:
        q = re.search(r"- 问题：(.*)", e)
        c = re.search(r"- 建议修正：(.*)", e)
        tier = re.search(r"- 置信分层：(.*)", e)
        merged_lines.append(
            f"- 问题：{(q.group(1).strip() if q else '未知')}｜置信分层：{(tier.group(1).strip() if tier else 'unknown')}"
        )
        if c:
            merged_lines.append(f"  - 建议修正：{c.group(1).strip()}")

    _append_text(main_doc, "\n".join(merged_lines) + "\n")

    archive_dir = patch_file.parent / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = archive_dir / f"{patch_file.stem}.{datetime.now().strftime('%Y%m%d%H%M%S')}{patch_file.suffix}"
    patch_file.replace(archived)

    if recreate_header:
        topic = patch_file.parent.parent.name if patch_file.parent.parent else "综合"
        tier_name = patch_file.parent.name if patch_file.parent else "medium"
        _ensure_patch_header(patch_file, topic=topic, tier=tier_name)
    return True


def _iter_active_patch_files() -> list[tuple[str, Path]]:
    active: list[tuple[str, Path]] = []
    _migrate_legacy_patches_once()
    if not PATCH_ROOT.exists():
        return active
    for root_folder in [x for x in PATCH_ROOT.iterdir() if x.is_dir()]:
        for p in root_folder.rglob(f"*{PATCH_DOC_NAME}"):
            if not p.is_file():
                continue
            if "archive" in p.parts:
                continue
            rel = p.relative_to(PATCH_ROOT).as_posix()
            active.append((rel, p))
    return active


def _patch_topic_tier_from_rel(rel_path: str) -> tuple[str, str]:
    parts = Path(rel_path).parts
    # {scope}/{topic}/{tier}/{file}
    if len(parts) >= 4:
        return parts[1], parts[2]
    return "综合", "unknown"


def _build_patch_stats() -> dict:
    files = _iter_active_patch_files()
    topic_stats: dict[str, dict[str, int]] = {}
    total_entries = 0
    total_files = 0

    for rel, p in files:
        entries = _count_patch_entries(p)
        if entries <= 0:
            continue
        total_files += 1
        total_entries += entries
        topic, tier = _patch_topic_tier_from_rel(rel)
        bucket = topic_stats.setdefault(topic, {"entries": 0, "files": 0, "high": 0, "medium": 0, "low": 0})
        bucket["entries"] += entries
        bucket["files"] += 1
        if tier in {"high", "medium", "low"}:
            bucket[tier] += entries

    audits = _read_jsonl(PATCH_AUDIT_LOG)
    dedup_count = sum(1 for a in audits if a.get("status") == "deduplicated")
    merged_count = sum(1 for a in audits if a.get("status") == "merged")
    considered = sum(1 for a in audits if a.get("status") in {"applied", "merged", "deduplicated"})
    dedup_rate = (dedup_count / considered) if considered else 0.0

    return {
        "total_active_patch_files": total_files,
        "total_active_patch_entries": total_entries,
        "deduplicated_count": dedup_count,
        "merged_count": merged_count,
        "dedup_rate": round(dedup_rate, 4),
        "topic_stats": topic_stats,
    }


def _review_and_merge_all_patches(cleanup: bool = True) -> dict:
    merged_files = 0
    merged_entries = 0
    for rel, p in _iter_active_patch_files():
        entries = _count_patch_entries(p)
        if entries <= 0:
            continue
        ok = _merge_patch_into_main_doc(rel, p, recreate_header=not cleanup)
        if ok:
            merged_files += 1
            merged_entries += entries
            _append_jsonl(
                PATCH_AUDIT_LOG,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "status": "review-merged",
                    "patch_path": rel,
                    "entries": entries,
                },
            )

    try:
        service.ingest("./data/documents")
        synced = True
    except Exception:
        synced = False

    return {
        "merged_files": merged_files,
        "merged_entries": merged_entries,
        "cleanup": cleanup,
        "synced": synced,
    }


def _maybe_apply_feedback_patch(
    *,
    question: str,
    answer: str,
    confidence_note: str,
    rating: int,
    corrected_answer: str,
    comment: str,
) -> tuple[bool, str | None, str | None, str]:
    trigger = (rating < 0) or bool(corrected_answer.strip())
    if not trigger:
        _append_jsonl(
            PATCH_AUDIT_LOG,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "status": "skipped",
                "question": question,
            },
        )
        return False, None, None, "skipped"

    topic = _feedback_topic(question, corrected_answer, comment)
    tier = _feedback_confidence_tier(confidence_note, rating, corrected_answer)
    carrier = _resolve_carrier_code(f"{question}\n{corrected_answer}")
    folder = carrier if carrier else "airport"
    month_key = datetime.now().strftime("%Y-%m")
    rel_path = f"{folder}/{topic}/{tier}/{month_key}-{PATCH_DOC_NAME}"
    full_path = _safe_patch_path(rel_path)

    fingerprint = _feedback_fingerprint(question, corrected_answer, topic)
    if fingerprint in _existing_patch_fingerprints():
        _append_jsonl(
            PATCH_AUDIT_LOG,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "status": "deduplicated",
                "question": question,
                "topic": topic,
                "tier": tier,
                "patch_path": rel_path,
                "fingerprint": fingerprint,
            },
        )
        try:
            iterated = service.ask(question)
            return False, rel_path, iterated.answer, "deduplicated"
        except Exception:
            return False, rel_path, None, "deduplicated"

    _ensure_patch_header(full_path, topic=topic, tier=tier)

    patch_md = _build_feedback_patch_markdown(
        question=question,
        answer=answer,
        confidence_note=confidence_note,
        rating=rating,
        corrected_answer=corrected_answer,
        comment=comment,
    )
    _append_text(full_path, patch_md)
    _append_jsonl(
        PATCH_REGISTRY_LOG,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "fingerprint": fingerprint,
            "question": question,
            "topic": topic,
            "tier": tier,
            "patch_path": rel_path,
        },
    )

    merged = False
    if full_path.exists() and full_path.stat().st_size >= PATCH_MAX_BYTES:
        merged = _merge_patch_into_main_doc(rel_path, full_path)
    elif _count_patch_entries(full_path) >= PATCH_MERGE_ENTRY_THRESHOLD:
        merged = _merge_patch_into_main_doc(rel_path, full_path)

    try:
        service.ingest("./data/documents")
        iterated = service.ask(question)
        status = "merged" if merged else "applied"
        _append_jsonl(
            PATCH_AUDIT_LOG,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "status": status,
                "question": question,
                "topic": topic,
                "tier": tier,
                "patch_path": rel_path,
                "fingerprint": fingerprint,
            },
        )
        return True, rel_path, iterated.answer, status
    except Exception:
        status = "merged" if merged else "applied"
        _append_jsonl(
            PATCH_AUDIT_LOG,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "status": status,
                "question": question,
                "topic": topic,
                "tier": tier,
                "patch_path": rel_path,
                "fingerprint": fingerprint,
                "sync_error": True,
            },
        )
        return True, rel_path, None, status


def _record_uncovered_question(
    *,
    question: str,
    answer_id: str,
    confidence_note: str,
    reason: str,
    rating: int | None = None,
) -> None:
    payload = {
        "event_id": str(uuid4()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "answer_id": answer_id,
        "question": question,
        "confidence_note": confidence_note,
        "reason": reason,
        "rating": rating,
    }
    _append_jsonl(UNCOVERED_LOG, payload)


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


@app.get("/admin/patches")
def admin_patches_home() -> FileResponse:
    page_file = STATIC_DIR / "patches.html"
    if not page_file.exists():
        raise HTTPException(status_code=404, detail="Patches admin page not found")
    return FileResponse(page_file)


@app.get("/admin/ocr-review")
def admin_ocr_review_home() -> FileResponse:
    page_file = STATIC_DIR / "ocr_review.html"
    if not page_file.exists():
        raise HTTPException(status_code=404, detail="OCR review page not found")
    return FileResponse(page_file)


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


def _safe_ocr_sidecar_path(relative_path: str) -> Path:
    full = _safe_doc_path(relative_path)
    rel = full.relative_to(DOC_ROOT).as_posix().lower()
    if not rel.endswith(".ocr.md"):
        raise HTTPException(status_code=400, detail="path must be an .ocr.md sidecar file")
    return full


def _safe_patch_path(relative_path: str) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute():
        raise HTTPException(status_code=400, detail="patch path must be relative")
    candidate = (PATCH_ROOT / rel).resolve()
    if PATCH_ROOT not in candidate.parents and candidate != PATCH_ROOT:
        raise HTTPException(status_code=400, detail="invalid patch path")
    return candidate


def _migrate_legacy_patches_once() -> None:
    global _LEGACY_PATCH_MIGRATED
    if _LEGACY_PATCH_MIGRATED:
        return
    _LEGACY_PATCH_MIGRATED = True

    if not DOC_ROOT.exists():
        return
    PATCH_ROOT.mkdir(parents=True, exist_ok=True)

    for scope_dir in [x for x in DOC_ROOT.iterdir() if x.is_dir()]:
        legacy = scope_dir / "patches"
        if not legacy.exists():
            continue
        target_scope = PATCH_ROOT / scope_dir.name
        target_scope.mkdir(parents=True, exist_ok=True)
        for item in legacy.iterdir():
            dest = target_scope / item.name
            if dest.exists():
                continue
            item.replace(dest)
        try:
            legacy.rmdir()
        except OSError:
            pass


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


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


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

            ocr_text_path = None
            ocr_chars = 0
            if _is_image_file(target):
                ocr_text = extract_text_from_image(target).strip()
                if ocr_text:
                    ocr_target = _safe_doc_path(f"{folder}/{fname}.ocr.md")
                    ocr_target.write_text(
                        f"# OCR 文本：{fname}\n\n{ocr_text}\n",
                        encoding="utf-8",
                    )
                    ocr_text_path = ocr_target.relative_to(DOC_ROOT).as_posix()
                    ocr_chars = len(ocr_text)

            created.append(
                {
                    "path": target.relative_to(DOC_ROOT).as_posix(),
                    "carrier": carrier,
                    "binary": not is_text,
                    "ocr_text_path": ocr_text_path,
                    "ocr_chars": ocr_chars,
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


@app.get("/admin/ocr-review/items")
def admin_list_ocr_review_items(limit: int = Query(200, ge=1, le=1000)) -> dict:
    DOC_ROOT.mkdir(parents=True, exist_ok=True)
    items = []
    for p in DOC_ROOT.rglob("*.ocr.md"):
        if not p.is_file():
            continue
        rel = p.relative_to(DOC_ROOT).as_posix()
        if any(part.startswith(".") for part in rel.split("/")):
            continue

        source_rel = rel[:-7]
        source_path = _safe_doc_path(source_rel)
        source_exists = source_path.exists() and source_path.is_file()

        stat = p.stat()
        source_updated_at = int(source_path.stat().st_mtime) if source_exists else None
        stale = bool(source_exists and source_updated_at and source_updated_at > int(stat.st_mtime))

        chars = 0
        try:
            chars = len(p.read_text(encoding="utf-8").strip())
        except UnicodeDecodeError:
            try:
                chars = len(p.read_bytes().decode("gb18030", errors="ignore").strip())
            except Exception:
                chars = 0

        items.append(
            {
                "ocr_path": rel,
                "source_path": source_rel,
                "source_exists": source_exists,
                "source_media_type": _guess_media_type(source_path) if source_exists else None,
                "source_preview_url": f"/admin/docs/raw?path={source_rel}",
                "ocr_content_url": f"/admin/docs/content?path={rel}",
                "chars": chars,
                "updated_at": int(stat.st_mtime),
                "source_updated_at": source_updated_at,
                "stale": stale,
            }
        )

    items.sort(key=lambda x: (not x["stale"], -x["updated_at"], x["ocr_path"]))
    return {
        "root": str(DOC_ROOT),
        "total": len(items),
        "items": items[:limit],
    }


@app.put("/admin/ocr-review/content")
def admin_update_ocr_review_content(path: str = Query(..., min_length=1), req: AdminUpdateDocRequest = None) -> dict:
    if req is None:
        raise HTTPException(status_code=400, detail="request body required")
    full = _safe_ocr_sidecar_path(path)
    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail="ocr sidecar not found")
    full.write_text(req.content or "", encoding="utf-8")
    sync_result = _sync_if_needed(req.auto_sync)
    return {"path": full.relative_to(DOC_ROOT).as_posix(), **sync_result}


@app.get("/admin/docs/raw")
def admin_get_doc_raw(path: str = Query(..., min_length=1)) -> FileResponse:
    full = _safe_doc_path(path)
    if not full.exists() or not full.is_file():
        raise HTTPException(status_code=404, detail="document not found")
    return FileResponse(
        path=full,
        media_type=_guess_media_type(full),
        filename=full.name,
        content_disposition_type="inline",
    )


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


@app.post("/flight/realtime")
def flight_realtime(req: FlightRealtimeRequest) -> dict:
    question = (req.question or "").strip()
    flight_no = (req.flight_no or "").strip() or None
    if not question and flight_no:
        question = f"{flight_no} 实时航班状态"
    if not question:
        raise HTTPException(status_code=400, detail="question 或 flight_no 至少提供一个")

    try:
        result = query_realtime_flight(question=question, flight_no=flight_no)
        if result is None:
            raise HTTPException(status_code=404, detail="未识别为实时航班查询问题")
        answer_text, card = result
        return {
            "question": question,
            "answer": answer_text,
            "confidence_note": "realtime-flight",
            "realtime_flight": card.model_dump(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"实时航班查询失败: {exc}") from exc


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        realtime_result = query_realtime_flight(question=req.question)
        if realtime_result is not None:
            answer_text, card = realtime_result
            return AskResponse(
                answer_id=str(uuid4()),
                question=req.question,
                answer=answer_text,
                citations=[],
                confidence_note="realtime-flight",
                realtime_flight=card,
            )

        result = service.ask(req.question, req.top_k)
        answer_id = str(uuid4())
        result.answer_id = answer_id

        if result.confidence_note in {"low-confidence", "index-empty"}:
            _record_uncovered_question(
                question=req.question,
                answer_id=answer_id,
                confidence_note=result.confidence_note,
                reason="auto-low-confidence",
            )

        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/feedback", response_model=AnswerFeedbackResponse)
def submit_feedback(req: AnswerFeedbackRequest) -> AnswerFeedbackResponse:
    feedback_id = str(uuid4())
    answer_id = req.answer_id or str(uuid4())
    corrected = (req.corrected_answer or "").strip()

    payload = {
        "feedback_id": feedback_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "answer_id": answer_id,
        "question": req.question,
        "answer": req.answer,
        "confidence_note": req.confidence_note,
        "rating": req.rating,
        "corrected_answer": corrected,
        "comment": (req.comment or "").strip(),
    }
    _append_jsonl(ANSWER_FEEDBACK_LOG, payload)

    patch_applied, patch_path, iterated_answer, patch_status = _maybe_apply_feedback_patch(
        question=req.question,
        answer=req.answer,
        confidence_note=req.confidence_note,
        rating=req.rating,
        corrected_answer=corrected,
        comment=(req.comment or "").strip(),
    )

    if req.rating < 0 or corrected:
        reason = "user-dislike" if req.rating < 0 else "user-correction"
        _record_uncovered_question(
            question=req.question,
            answer_id=answer_id,
            confidence_note=req.confidence_note,
            reason=reason,
            rating=req.rating,
        )

    return AnswerFeedbackResponse(
        status="ok",
        feedback_id=feedback_id,
        patch_applied=patch_applied,
        patch_status=patch_status,
        patch_path=patch_path,
        iterated_answer=iterated_answer,
    )


@app.get("/admin/patches/stats")
def admin_patch_stats() -> dict:
    return _build_patch_stats()


@app.post("/admin/patches/review-merge")
def admin_patch_review_merge(cleanup: bool = True) -> dict:
    result = _review_and_merge_all_patches(cleanup=cleanup)
    return {
        "status": "ok",
        **result,
        "stats": _build_patch_stats(),
    }


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
