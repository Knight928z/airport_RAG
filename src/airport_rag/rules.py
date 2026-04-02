from __future__ import annotations

import re
from dataclasses import dataclass

from .vector_store import RetrievedChunk


@dataclass
class RuleResult:
    answer: str
    note: str = "rule-based"
    evidence_chunk_ids: list[str] | None = None


def detect_question_language(question: str) -> str:
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", question or ""))
    letters = re.findall(r"[A-Za-z]", question or "")
    if letters and not has_cjk:
        return "en"
    if has_cjk:
        return "zh"
    return "zh"


def build_refund_rule_answer(question: str, retrieved: list[RetrievedChunk]) -> RuleResult | None:
    if not _is_refund_question(question):
        return None

    best_item: RetrievedChunk | None = None
    best_sentence = ""
    best_score = -1

    for item in retrieved:
        text = item.text or ""
        sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?])|\n+", text) if s.strip()]
        for sent in sentences:
            score = _refund_sentence_score(sent)
            if score > best_score:
                best_score = score
                best_sentence = sent
                best_item = item

    if best_item is None or best_score < 2:
        return None

    lang = detect_question_language(question)
    if lang == "en":
        conclusion = "Conclusion: Voluntary cancellation is generally not refundable in this scenario."
        evidence_header = "Evidence:"
        action = "Recommendation: Please verify whether your case belongs to airline-initiated flight change/cancellation or other explicit refundable exceptions."
        risk = "Risk note: Final outcome depends on latest airline policy and ticket conditions."
        translated_hint = _english_hint_for_refund(best_sentence)
    else:
        conclusion = "结论：自愿取消航班在该场景通常不能退款（不属于可退款例外情形）。"
        evidence_header = "依据："
        action = "执行建议：请核对是否属于航司变更/取消、直飞美国24小时内取消等明确可退例外条款。"
        risk = "风险提示：最终以航司最新运输条件与客票规则为准。"
        translated_hint = _chinese_hint_for_refund(best_sentence)

    evidence_line = f"- [1] {best_sentence}（来源：{best_item.source}）"
    if translated_hint:
        evidence_line += f"\n  释义：{translated_hint}"

    answer = "\n".join(
        [
            conclusion,
            evidence_header,
            evidence_line,
            action,
            risk,
        ]
    )

    return RuleResult(answer=answer, evidence_chunk_ids=[best_item.chunk_id])


def _is_refund_question(question: str) -> bool:
    q = (question or "").lower()
    zh_keywords = ["退款", "退票", "能退", "可退", "能否退", "自愿取消", "取消航班"]
    en_keywords = ["refund", "refundable", "cancel", "voluntarily", "cancellation"]
    return any(k in q for k in zh_keywords) or any(k in q for k in en_keywords)


def _refund_sentence_score(sentence: str) -> int:
    s = sentence.lower()
    score = 0
    if "refund" in s or "退款" in s or "退票" in s:
        score += 2
    if "not entitled" in s or "not refundable" in s or "不予退款" in s or "不享有退款" in s:
        score += 3
    if "voluntarily cancel" in s or "自愿取消" in s or "cancel your flight" in s:
        score += 3
    if "6.3" in s:
        score += 1
    return score


def _chinese_hint_for_refund(sentence: str) -> str:
    s = sentence.lower()
    if "not entitled" in s and ("voluntarily cancel" in s or "cancel your flight" in s):
        return "该条款明确：若旅客自愿取消航班，通常不享有已购产品退款。"
    if "not entitled" in s:
        return "该条款明确属于不予退款情形。"
    if "voluntarily cancel" in s:
        return "该条款指出自愿取消航班属于不退款条件。"
    return ""


def _english_hint_for_refund(sentence: str) -> str:
    s = sentence.lower()
    if "not entitled" in s and ("voluntarily cancel" in s or "cancel your flight" in s):
        return "This clause indicates that voluntary flight cancellation is generally non-refundable."
    if "not entitled" in s:
        return "This clause states that the case is not eligible for refund."
    return ""


def localize_answer_text(text: str, lang: str) -> str:
    if not text:
        return text
    if lang != "en":
        return text

    en_map = {
        "结论：": "Conclusion: ",
        "依据：": "Evidence:\n",
        "执行建议：": "Recommendation: ",
        "风险提示：": "Risk note: ",
        "根据文档，": "According to the document, ",
        "根据现有文档可直接确认：": "Based on current policy text, ",
        "可联系的客服热线/电话为：": "Available customer service hotline/phone: ",
        "当前知识库未检索到可直接回答该问题的业务证据": "The current knowledge base does not contain direct evidence for this question",
        "请补充更直接条款": "Please provide more direct policy clauses",
        "需人工复核": "manual review is required",
    "请按原文条款逐项核验，如涉及身份/证件条件请同时核对适用范围。": "Please verify each step against the original policy text and confirm identity/document applicability.",
    "若政策有更新或旅客身份属于特殊类型，请以现场部门最新规定为准。": "If policy updates apply or the passenger belongs to a special category, follow the latest on-site authority requirements.",
        "机场": "airport",
        "航司": "airline",
        "行李": "baggage",
        "托运": "checked baggage",
        "随身": "carry-on",
        "中转": "transit",
        "海关": "customs",
        "边检": "immigration",
        "客服": "customer service",
        "热线": "hotline",
        "孕妇": "pregnant passenger",
        "可乘机": "may travel by air",
    }

    localized = text
    for src in sorted(en_map.keys(), key=len, reverse=True):
        localized = localized.replace(src, en_map[src])
    return localized


def build_rule_based_answer(question: str, retrieved: list[RetrievedChunk]) -> RuleResult | None:
    comparison = _build_comparison_answer(question, retrieved)
    if comparison is not None:
        return comparison

    contact = _build_contact_answer(question, retrieved)
    if contact is not None:
        return contact

    pregnancy = _build_pregnancy_answer(question, retrieved)
    if pregnancy is not None:
        return pregnancy

    generic_numeric = build_numeric_fact_answer(question, retrieved)
    if generic_numeric is not None:
        return generic_numeric

    battery = _build_battery_answer(question, retrieved)
    if battery is not None:
        return battery

    stay_duration = _build_stay_duration_answer(question, retrieved)
    if stay_duration is not None:
        return stay_duration

    infant = _build_infant_ticket_answer(question, retrieved)
    if infant is not None:
        return infant

    return None


def build_pregnancy_answer(question: str, retrieved: list[RetrievedChunk]) -> RuleResult | None:
    return _build_pregnancy_answer(question, retrieved)


def extract_best_fact_sentence(question: str, retrieved: list[RetrievedChunk]) -> tuple[str, RetrievedChunk | None]:
    return _extract_best_fact_sentence(question, retrieved)


def is_contact_question(question: str) -> bool:
    return _is_contact_question(question)


def is_numeric_question(question: str) -> bool:
    return _is_numeric_question(question)


def is_duration_question(question: str) -> bool:
    return _is_duration_question(question)


def has_duration_fact(text: str) -> bool:
    return _has_duration_fact(text)


def is_fee_question(question: str) -> bool:
    return _is_fee_question(question)


def build_numeric_fact_answer(question: str, retrieved: list[RetrievedChunk]) -> RuleResult | None:
    if not _is_numeric_question(question):
        return None

    q_norm = _normalize_for_matching(question)
    anchors = _extract_anchor_terms(q_norm)
    content_tokens = _required_content_tokens(q_norm)
    required_tokens = _required_intent_tokens(q_norm)
    cue_groups = _expected_answer_cue_groups(question)
    fee_question = _is_fee_question(question)
    min_content_hits = 1 if required_tokens else 2
    best_item: RetrievedChunk | None = None
    best_sentence = ""
    best_score = -1.0

    for item in retrieved:
        text = _normalize_for_matching(item.text)
        sentences = [s.strip() for s in re.split(r"(?<=[。！？；;.!?])|\n+", text) if s.strip()]
        for sent in sentences:
            if not _has_numeric_fact(sent):
                continue
            if cue_groups and not _matches_cue_groups(sent, cue_groups):
                continue
            if fee_question and not any(k in sent for k in ["费用", "收费", "元", "人民币", "美元", "票价", "价格"]):
                continue
            req_hits = sum(1 for tok in required_tokens if tok in sent)
            if required_tokens and req_hits == 0:
                continue
            content_hits = sum(1 for tok in content_tokens if tok in sent)
            if content_tokens and content_hits < min_content_hits:
                continue
            anchor_hits = sum(1 for a in anchors if a in sent)
            sim = _bigram_similarity(_char_bigrams(q_norm), _char_bigrams(sent))
            score = req_hits * 5 + content_hits * 2 + anchor_hits * 2 + sim
            if score > best_score:
                best_score = score
                best_item = item
                best_sentence = sent

    if best_item is None or best_score < 1.0:
        return None

    q_join = _normalize_for_matching(f"{question} {best_item.source} {best_sentence}")
    cash_intent = any(k in q_join for k in ["现金", "现钞", "人民币", "美元", "海关"])
    border_intent = any(k in q_join for k in ["边检", "边防", "停留", "通行证", "入境"])

    recommendation = "执行建议：请按该数值阈值执行对应业务流程，并对照原文条款复核适用条件。"
    risk_note = "风险提示：如遇政策更新或口径差异，请以发布方最新公告与现场执行标准为准。"
    if cash_intent:
        recommendation = "执行建议：请按该数值阈值判断是否需要海关申报。"
        risk_note = "风险提示：如遇政策更新或币种换算差异，请以海关最新公告为准。"
    elif border_intent:
        recommendation = "执行建议：请同时核验证件类型、停留事由及口岸适用条件。"
        risk_note = "风险提示：边检政策可能动态调整，请以国家移民管理部门最新规定为准。"

    answer = (
        "结论：根据现有文档可直接确认："
        f"{best_sentence}\n"
        "依据：\n"
        f"- [1] {best_sentence}（来源：{best_item.source}）\n"
        f"{recommendation}\n"
        f"{risk_note}"
    )
    return RuleResult(answer=answer, evidence_chunk_ids=[best_item.chunk_id])


def build_factoid_answer(question: str, retrieved: list[RetrievedChunk]) -> RuleResult | None:
    if not retrieved:
        return None

    sentence, item = _extract_best_fact_sentence(question, retrieved)
    if not sentence or item is None:
        return None

    if _is_contact_question(question):
        if not _extract_contact_numbers(sentence):
            return None

    if _is_fee_question(question):
        if not re.search(r"\d", sentence):
            return None

    if _is_duration_question(question) and not _has_duration_fact(sentence):
        return None

    conclusion = f"根据文档，{sentence}"
    q_lower = question.lower()
    if "是否" in question or q_lower.endswith("吗？") or q_lower.endswith("吗"):
        if any(k in sentence for k in ["免填写", "无需填写", "可免"]):
            conclusion = f"根据文档，存在免填情形：{sentence}"
        elif any(k in sentence for k in ["应", "须", "需要"]):
            conclusion = f"需要。{sentence}"

    lines = [
        f"结论：{conclusion}",
        "依据：",
        f"- [1] {sentence}（来源：{item.source}）",
        "执行建议：请按原文条款逐项核验，如涉及身份/证件条件请同时核对适用范围。",
        "风险提示：若政策有更新或旅客身份属于特殊类型，请以现场部门最新规定为准。",
    ]
    return RuleResult(
        answer="\n".join(lines),
        note="grounded-factoid",
        evidence_chunk_ids=[item.chunk_id],
    )


def _expand_cross_lingual_query(question: str) -> str:
    q = question or ""
    q_lower = q.lower()

    zh_to_en = {
        "行李": "baggage luggage",
        "托运": "check-in checked",
        "随身": "carry-on cabin",
        "退票": "refund",
        "改签": "change rebook",
        "值机": "check-in",
        "中转": "transit transfer",
        "海关": "customs",
        "边检": "immigration border control",
        "客服": "customer service hotline",
        "热线": "hotline contact",
        "航班": "flight",
        "孕妇": "pregnant passenger",
    }
    en_to_zh = {
        "baggage": "行李",
        "luggage": "行李",
        "checked": "托运",
        "carry": "随身",
        "refund": "退票",
        "change": "改签",
        "rebook": "改签",
        "check-in": "值机",
        "transit": "中转",
        "transfer": "中转",
        "customs": "海关",
        "immigration": "边检",
        "hotline": "客服热线",
        "customer service": "客服",
        "flight": "航班",
        "pregnant": "孕妇",
    }

    extra: list[str] = []
    for zh, en in zh_to_en.items():
        if zh in q:
            extra.append(en)
    for en, zh in en_to_zh.items():
        if en in q_lower:
            extra.append(zh)

    return q if not extra else f"{q} {' '.join(extra)}"


def _extract_best_fact_sentence(question: str, retrieved: list[RetrievedChunk]) -> tuple[str, RetrievedChunk | None]:
    expanded_q = _expand_cross_lingual_query(question)
    q_norm = _normalize_for_matching(expanded_q)
    anchors = _extract_anchor_terms(q_norm)
    content_tokens = _required_content_tokens(q_norm)
    cue_groups = _expected_answer_cue_groups(question)
    subject_terms = _subject_terms(question)
    salient_terms, salient_long_terms = _salient_terms(question)
    q_bigrams = _char_bigrams(q_norm)

    best_score = -1.0
    best_sentence = ""
    best_item: RetrievedChunk | None = None

    for item in retrieved:
        text = _normalize_for_matching(item.text)
        sentences = [s.strip() for s in re.split(r"(?<=[。！？；;.!?])|\n+", text) if s.strip()]
        for sent in sentences:
            if cue_groups and not _matches_cue_groups(sent, cue_groups):
                continue
            subject_hits = sum(1 for t in subject_terms if t in sent)
            if subject_terms and subject_hits == 0:
                continue
            salient_hits = sum(1 for t in salient_terms if t in sent)
            salient_long_hits = sum(1 for t in salient_long_terms if t in sent)
            if salient_long_terms and salient_long_hits == 0 and salient_hits < 2:
                continue
            content_hits = sum(1 for t in content_tokens if t in sent)
            if content_tokens and content_hits == 0:
                continue
            anchor_hits = sum(1 for a in anchors if a in sent)
            sim = _bigram_similarity(q_bigrams, _char_bigrams(sent))
            policy_hits = sum(1 for k in ["应", "须", "可", "不得", "禁止", "持用", "填写", "选择", "通行证", "入境卡"] if k in sent)
            score = (
                subject_hits * 2.8
                + salient_hits * 3.0
                + salient_long_hits * 2.5
                + content_hits * 2.5
                + anchor_hits * 2
                + policy_hits * 0.7
                + sim
            )
            if score > best_score:
                best_score = score
                best_sentence = sent
                best_item = item

    if best_score < 1.2:
        return "", None
    return best_sentence, best_item


def _build_battery_answer(question: str, retrieved: list[RetrievedChunk]) -> RuleResult | None:
    q_lower = question.lower()
    has_wh_token = bool(re.search(r"\b\d+(?:\.\d+)?\s*wh\b|\bwh\b", q_lower))
    if not any(k in q_lower for k in ["充电宝", "锂电池", "额定能量"]) and not has_wh_token:
        return None

    evidence = [
        r for r in retrieved if any(k in f"{r.source} {r.text}".lower() for k in ["充电宝", "锂电池", "wh", "额定能量"])
    ]
    if not evidence:
        return None

    asked_wh = _extract_wh_value(question)
    text_all = "\n".join(r.text.lower() for r in evidence)

    forbid_over_100 = bool(re.search(r"超过\s*100\s*wh.{0,20}(禁止|严禁|不得|不能)", text_all))
    forbid_over_160 = bool(re.search(r"超过\s*160\s*wh.{0,20}(禁止|严禁|不得|不能)", text_all))
    allow_le_100 = bool(re.search(r"(小于|不超过|小于等于|≤).{0,6}100\s*wh", text_all))
    allow_100_160_with_approval = bool(
        re.search(r"超过\s*100\s*wh.{0,30}不超过\s*160\s*wh", text_all)
        and re.search(r"(航空公司).{0,12}(同意|批准)", text_all)
    )

    conclusion = "需人工复核。"
    if asked_wh is not None:
        if asked_wh > 160:
            conclusion = "不能。"
        elif asked_wh > 100:
            if forbid_over_100:
                conclusion = "不能。"
            elif allow_100_160_with_approval:
                conclusion = "有条件可以（需航空公司同意）。"
            elif allow_le_100:
                conclusion = "当前证据仅明确≤100Wh携带条件，>100Wh需按航司条款进一步确认。"
            elif forbid_over_160:
                conclusion = "通常需航空公司确认后方可判断，建议按现场规定执行。"
            else:
                conclusion = "需人工复核。"
        else:
            if allow_le_100 or forbid_over_100 or allow_100_160_with_approval:
                conclusion = "可以。"

    lines = [
        f"结论：针对{asked_wh if asked_wh is not None else '该'}Wh充电宝，{conclusion}",
        "依据：",
    ]
    for idx, item in enumerate(evidence[:2], start=1):
        span = _extract_battery_evidence_span(item.text)
        lines.append(f"- [{idx}] {span}（来源：{item.source}）")
    lines.extend(
        [
            "执行建议：以航空公司及安检现场最新规定为准，必要时提前向航司确认。",
            "风险提示：不同航司/机型及临时管控可能存在差异，需人工复核。",
        ]
    )
    return RuleResult(answer="\n".join(lines), evidence_chunk_ids=[item.chunk_id for item in evidence[:2]])


def _extract_battery_evidence_span(text: str, max_chars: int = 160) -> str:
    battery_keywords = ["充电宝", "锂电池", "额定能量", "wh", "瓦特小时"]
    sentences = [s.strip() for s in re.split(r"(?<=[。！？；;.!?])|\n+", text or "") if s.strip()]

    for sent in sentences:
        lowered = sent.lower()
        if any(k in lowered for k in battery_keywords):
            return sent[:max_chars]

    return _extract_relevant_span("充电宝 锂电池 额定能量 wh", text or "", max_chars=max_chars)


def _build_stay_duration_answer(question: str, retrieved: list[RetrievedChunk]) -> RuleResult | None:
    q = question.lower()
    if not (
        any(k in q for k in ["停留多久", "停留多长", "停留时间", "能停留多久", "可停留多久"])
        or ("停留" in q and "多久" in q)
    ):
        return None

    evidence = []
    for r in retrieved:
        joined = f"{r.source} {r.text}"
        if any(k in joined for k in ["外国籍", "港澳", "内地", "停留", "边防", "通行证"]):
            evidence.append(r)

    if not evidence:
        return None

    duration_days = None
    best_item: RetrievedChunk | None = None

    def _context_score(text: str) -> int:
        score = 0
        if "外国籍港澳居民" in text:
            score += 10
        if "港澳居民来往内地" in text:
            score += 8
        if "非中国籍" in text:
            score += 6
        if "每次停留" in text:
            score += 4
        if "过境" in text:
            score -= 5
        if "55国" in text:
            score -= 5
        return score

    ranked = sorted(evidence, key=lambda r: _context_score(r.text), reverse=True)
    for item in ranked:
        m = re.search(r"停留[^。；\n]{0,30}不?能?超过\s*(\d+)\s*天", item.text)
        if not m:
            m = re.search(r"每次停留[^。；\n]{0,20}(\d+)\s*天", item.text)
        if m:
            duration_days = int(m.group(1))
            best_item = item
            break

    if duration_days is None or best_item is None:
        return None

    lines = [
        f"结论：外国籍港澳居民来往内地，每次停留不能超过{duration_days}天。",
        "依据：",
    ]
    lines.append(f"- [1] {best_item.text[:170]}（来源：{best_item.source}）")
    lines.extend(
        [
            "执行建议：请同时核验所持证件类型及有效期，按边检现场规定办理。",
            "风险提示：如遇政策更新或个案限制，以国家移民管理部门最新规定为准。",
        ]
    )

    return RuleResult(answer="\n".join(lines), evidence_chunk_ids=[best_item.chunk_id])


def _build_infant_ticket_answer(question: str, retrieved: list[RetrievedChunk]) -> RuleResult | None:
    ticket_intent = any(k in question for k in ["婴儿票", "儿童票", "几岁", "周岁", "年龄"]) or (
        "票" in question and "岁" in question
    )
    if not ticket_intent:
        return None

    evidence = [
        r
        for r in retrieved
        if any(k in f"{r.source} {r.text}" for k in ["婴儿票", "儿童票", "周岁", "年龄", "起飞日期", "未满2周岁", "满2周岁"])
    ]
    if not evidence:
        return None

    asked_age = _extract_age_value(question)
    text_all = "\n".join(r.text for r in evidence)
    has_infant_rule = bool(re.search(r"(未满|小于|不足)\s*2\s*周岁.{0,20}婴儿票", text_all))
    has_child_rule = bool(re.search(r"(满|达到)\s*2\s*周岁.{0,20}(儿童票|儿童客票)", text_all))

    conclusion = "请按文档年龄分界购买对应票种。"
    if asked_age is not None and has_infant_rule:
        if asked_age < 2:
            conclusion = "应购买婴儿票。"
        elif has_child_rule:
            conclusion = "不应购买婴儿票，应购买儿童票。"
        else:
            conclusion = "不应购买婴儿票，建议按航司儿童/成人票规则办理。"

    lines = [
        f"结论：{conclusion}",
        "依据：",
    ]
    for idx, item in enumerate(evidence[:2], start=1):
        lines.append(f"- [{idx}] {item.text[:150]}（来源：{item.source}）")
    lines.extend(
        [
            "执行建议：以起飞日期当日年龄为准，在购票环节选择正确票种。",
            "风险提示：跨日航班或特殊政策可能有差异，请以航司规则为准。",
        ]
    )
    return RuleResult(answer="\n".join(lines), evidence_chunk_ids=[item.chunk_id for item in evidence[:2]])


def _extract_wh_value(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*wh", text.lower())
    if not m:
        return None
    return float(m.group(1))


def _extract_age_value(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*岁", text)
    if not m:
        return None
    return float(m.group(1))


def _tokenize(text: str) -> set[str]:
    lowered = text.lower()
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    words = re.findall(r"[A-Za-z0-9_]+", lowered)
    return set(cjk_chars + words)


def _query_terms(text: str) -> set[str]:
    lowered = text.lower()
    cjk_terms = re.findall(r"[\u4e00-\u9fff]{2,}", lowered)
    words = re.findall(r"[A-Za-z0-9_]{2,}", lowered)
    return set(cjk_terms + words)


def _extract_relevant_span(question: str, text: str, max_chars: int = 180) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned

    q_tokens = _tokenize(question)
    q_terms = _query_terms(question)
    sentences = [s.strip() for s in re.split(r"(?<=[。！？.!?])|\n+", cleaned) if s.strip()]
    clauses: list[str] = []
    for sent in sentences:
        clauses.extend([c.strip(" ，、；;:") for c in re.split(r"[，、；;：:]", sent) if c.strip()])

    if clauses:
        paired = [f"{clauses[i]}，{clauses[i + 1]}" for i in range(len(clauses) - 1)]
        candidates = clauses + paired
    else:
        candidates = sentences

    if not candidates:
        return cleaned[:max_chars]

    keywords = _critical_keywords(question)
    if keywords:
        keyword_candidates = [c for c in candidates if any(k in c.lower() for k in keywords)]
        if keyword_candidates:
            candidates = keyword_candidates

    def sent_score(sent: str) -> tuple[int, int, int]:
        sent_tokens = _tokenize(sent)
        token_overlap = len(q_tokens.intersection(sent_tokens))
        term_overlap = sum(1 for term in q_terms if term in sent)
        length_pref = -abs(len(sent) - min(60, max_chars))
        return term_overlap, token_overlap, length_pref

    ranked = sorted(candidates, key=sent_score, reverse=True)
    best = ranked[0]
    best_term, best_token, _ = sent_score(best)

    if best_term == 0 and best_token == 0:
        return cleaned[:max_chars]
    return best[:max_chars]


def _critical_keywords(question: str) -> set[str]:
    q = question.lower()
    if any(k in q for k in ["充电宝", "锂电池", "wh", "额定能量"]):
        return {"充电宝", "锂电池", "电池", "wh", "额定能量", "禁止", "不得", "可以", "同意"}
    if any(k in q for k in ["婴儿票", "儿童票", "周岁", "年龄", "几岁"]):
        return {"婴儿票", "儿童票", "周岁", "未满", "满", "起飞日期"}
    return set()


def _is_fee_question(question: str) -> bool:
    return any(k in question for k in ["费用", "收费", "多少钱", "金额", "票价", "价格"])


def _expected_answer_cue_groups(question: str) -> list[set[str]]:
    q = question.lower()
    groups: list[set[str]] = []

    subject_terms = _subject_terms(question)
    if subject_terms:
        groups.append(subject_terms)

    if any(k in q for k in ["什么时候", "多久", "几点", "提前", "时间", "关闭", "截止"]):
        groups.append({"小时", "分钟", "天", "提前", "截止", "关闭", "时", "点"})

    if _is_fee_question(question):
        groups.append({"费用", "收费", "元", "人民币", "美元", "票价", "价格"})

    if any(k in q for k in ["哪里", "在哪", "何处"]):
        groups.append({"在", "于", "位于", "柜台", "窗口", "区域", "航站楼", "大厅", "现场", "边检", "海关"})

    if any(k in q for k in ["领取", "获取", "拿", "在哪领"]):
        groups.append({"领取", "发放", "窗口", "柜台", "现场", "服务台", "办理点"})

    if any(k in q for k in ["证件", "证明", "材料", "通行证", "入境卡", "护照", "签证"]):
        groups.append({"证", "卡", "护照", "签证", "通行证", "身份证", "材料", "证明"})

    if _is_contact_question(question):
        groups.append({"电话", "热线", "客服", "联系", "号码", "955", "400", "+86"})

    if re.search(r"(能|可以|可否|是否).{0,6}吗", q) or "能带" in q:
        groups.append({"可", "可以", "不得", "禁止", "须", "应", "需要", "不能", "方可", "免"})

    return groups


def _subject_terms(question: str) -> set[str]:
    normalized = _normalize_for_matching(question)
    terms = _extract_anchor_terms(normalized)
    stop = {
        "请问",
        "如何",
        "怎么",
        "怎么办",
        "可以",
        "能够",
        "是否",
        "需要",
        "多少",
        "多久",
        "一般",
        "规定",
        "要求",
        "旅客",
        "国内",
        "国际",
        "入境",
        "出境",
        "离境",
        "进境",
    }
    return {t for t in terms if 2 <= len(t) <= 8 and t not in stop}


def _salient_terms(question: str) -> tuple[set[str], set[str]]:
    lowered = _normalize_for_matching(question)
    for token in [
        "请问",
        "吗",
        "么",
        "呢",
        "如何",
        "怎么",
        "怎么办",
        "需要",
        "是否",
        "可以",
        "能够",
        "一般",
        "哪些",
        "什么",
        "多少",
        "多久",
        "什么时候",
        "上限",
        "阈值",
    ]:
        lowered = lowered.replace(token, " ")

    base_tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]{2,}", lowered)
    stop = {
        "国内",
        "国际",
        "入境",
        "出境",
        "离境",
        "旅客",
        "规定",
        "要求",
        "流程",
    }

    all_terms: set[str] = set()
    long_terms: set[str] = set()
    for tok in base_tokens:
        if tok in stop:
            continue
        all_terms.add(tok)
        if 3 <= len(tok) <= 10:
            long_terms.add(tok)
        if re.fullmatch(r"[\u4e00-\u9fff]{5,}", tok):
            for n in (2, 3, 4):
                for i in range(0, len(tok) - n + 1):
                    sub = tok[i : i + n]
                    if sub not in stop:
                        all_terms.add(sub)

    return all_terms, long_terms


def _matches_cue_groups(text: str, cue_groups: list[set[str]]) -> bool:
    if not cue_groups:
        return True
    return all(any(c in text for c in group) for group in cue_groups if group)


def _is_contact_question(question: str) -> bool:
    q = question.lower()
    return any(
        k in q
        for k in [
            "热线",
            "电话",
            "联系电话",
            "联系方式",
            "客服电话",
            "客服",
            "号码",
            "怎么联系",
            "hotline",
            "customer service",
            "contact",
            "call center",
            "phone",
        ]
    )


def _is_compare_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["区别", "不同", "一样", "相同", "近似", "对比", "分别", "差异", "difference", "compare", "similar"])


def _source_scope_and_carrier(item: RetrievedChunk) -> tuple[str, str]:
    if item.doc_scope and item.doc_scope != "unknown":
        return item.doc_scope, item.carrier.upper()

    normalized = item.source.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]
    lowered = [p.lower() for p in parts]
    if "documents" in lowered:
        idx = lowered.index("documents")
        if idx + 1 < len(parts):
            folder = parts[idx + 1]
            if folder.lower() == "airport":
                return "airport", ""
            return "airline", folder.upper()
    return "unknown", ""


def _build_comparison_answer(question: str, retrieved: list[RetrievedChunk]) -> RuleResult | None:
    if not _is_compare_question(question):
        return None

    airport_item: RetrievedChunk | None = None
    airline_item: RetrievedChunk | None = None
    for item in retrieved:
        scope, _carrier = _source_scope_and_carrier(item)
        if scope == "airport" and airport_item is None:
            airport_item = item
        if scope == "airline" and airline_item is None:
            airline_item = item
        if airport_item and airline_item:
            break

    if not airport_item and not airline_item:
        return None

    lines = []
    evidence_ids: list[str] = []
    if airport_item and airline_item:
        lines.append("结论：该问题涉及机场与航司双重规则，存在差异可能，需分别遵循对应条款。")
    elif airport_item:
        lines.append("结论：当前仅检索到机场统一规则证据，暂未命中明确航司差异条款。")
    else:
        lines.append("结论：当前仅检索到航司规则证据，暂未命中机场统一规则条款。")

    lines.append("依据：")
    if airport_item:
        span = _extract_relevant_span(question, airport_item.text, max_chars=120)
        lines.append(f"- [机场规则] {span}（来源：{airport_item.source}）")
        evidence_ids.append(airport_item.chunk_id)
    if airline_item:
        span = _extract_relevant_span(question, airline_item.text, max_chars=120)
        lines.append(f"- [航司规则] {span}（来源：{airline_item.source}）")
        evidence_ids.append(airline_item.chunk_id)

    lines.append("执行建议：若同一事项存在冲突，优先满足更严格限制，并以承运航司与机场现场最新要求为准。")
    lines.append("风险提示：不同航司/航线可能存在近似但不完全一致的细则，办理前请二次核验。")
    return RuleResult(answer="\n".join(lines), evidence_chunk_ids=evidence_ids)


def _is_pregnancy_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["孕妇", "怀孕", "妊娠", "产后", "分娩"])


def _question_mentions_specific_carrier(question: str) -> bool:
    q = question.lower()
    aliases = [
        "南航",
        "中国南方航空",
        "南方航空",
        "东航",
        "国航",
        "春秋",
        "春秋航空",
        "阿联酋",
        "阿联酋航空",
        "emirates",
    ]
    if any(a in q for a in aliases):
        return True
    code_match = re.search(r"(?<![A-Za-z0-9])([A-Za-z0-9]{2})(?![A-Za-z0-9])", question)
    if not code_match:
        return False
    return code_match.group(1).upper() != "WH"


def _build_pregnancy_answer(question: str, retrieved: list[RetrievedChunk]) -> RuleResult | None:
    if not _is_pregnancy_question(question):
        return None

    cue_keywords = ["孕妇", "妊娠", "分娩", "先兆", "流产", "产后", "医学证明", "不宜乘机", "限制"]
    best_item: RetrievedChunk | None = None
    best_sentences: list[str] = []
    best_score = -1.0

    for item in retrieved:
        text = _normalize_for_matching(item.text)
        sentences = [s.strip() for s in re.split(r"(?<=[。！？；;.!?])|\n+", text) if s.strip()]
        matched = [s for s in sentences if any(k in s for k in cue_keywords)]
        if not matched:
            continue
        score = sum(sum(1 for k in cue_keywords if k in s) for s in matched[:3])
        if score > best_score:
            best_score = score
            best_item = item
            best_sentences = matched[:2]

    if best_item is None or not best_sentences:
        if not _question_mentions_specific_carrier(question):
            lines = [
                "结论：孕妇旅客是否可乘机需按具体承运航司规则判断，不能一概而论。",
                "依据：当前检索结果未命中可直接裁决的统一条款。",
                "执行建议：请补充航司名称/代码（如 MU、CZ、9C）后再查询，以获取对应承运限制。",
                "风险提示：孕期、并发症、先兆流产及产后时长等因素可能导致限制承运，需以值机审核为准。",
            ]
            return RuleResult(answer="\n".join(lines), evidence_chunk_ids=[])
        return None

    key_line = "；".join(best_sentences)
    conclusion = "根据现有航司条款，孕妇旅客可乘机但需满足特定健康与时限条件，特殊情形可能被限制承运。"
    lines = [
        f"结论：{conclusion}",
        "依据：",
        f"- [1] {key_line}（来源：{best_item.source}）",
        "执行建议：请对照孕周、是否存在并发症/先兆流产、产后时间及是否需医学证明逐项确认。",
        "风险提示：孕妇乘机属于高风险场景，最终以承运航司与现场值机审核结论为准。",
    ]
    return RuleResult(answer="\n".join(lines), evidence_chunk_ids=[best_item.chunk_id])


def _extract_contact_numbers(text: str) -> list[str]:
    patterns = [
        r"\+?\d{1,3}[- ]?\d{7,12}",
        r"\b\d{5,6}\b",
        r"\b400[- ]?\d{3,4}[- ]?\d{3,4}\b",
        r"\b\d{3,4}[- ]?\d{7,8}\b",
    ]
    found: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            num = re.sub(r"\s+", "", match.group(0).strip())
            if num not in found:
                found.append(num)
    return found


def _build_contact_answer(question: str, retrieved: list[RetrievedChunk]) -> RuleResult | None:
    if not _is_contact_question(question):
        return None

    if not _question_mentions_specific_carrier(question):
        return None

    best_item: RetrievedChunk | None = None
    best_numbers: list[str] = []
    best_score = -1

    for item in retrieved:
        joined = _normalize_for_matching(f"{item.source} {item.text}")
        cue_score = sum(1 for kw in ["客服", "热线", "电话", "联系", "号码"] if kw in joined)
        numbers = _extract_contact_numbers(joined)
        if not numbers:
            continue
        score = cue_score * 2 + len(numbers) * 3
        if score > best_score:
            best_score = score
            best_numbers = numbers
            best_item = item

    if best_item is None or not best_numbers:
        return None

    numbers_text = "、".join(best_numbers)
    lines = [
        f"结论：可联系的客服热线/电话为：{numbers_text}。",
        "依据：",
        f"- [1] {best_item.text[:170]}（来源：{best_item.source}）",
        "执行建议：优先拨打文档中明确列出的官方热线，跨境场景注意区号/国家码。",
        "风险提示：客服联系方式可能更新，请以航司官网与最新公告为准。",
    ]
    return RuleResult(answer="\n".join(lines), evidence_chunk_ids=[best_item.chunk_id])


def _is_numeric_question(question: str) -> bool:
    if _is_contact_question(question):
        return False
    return any(k in question for k in ["多少", "多久", "几", "最多", "上限", "不能超过", "不超过", "限额", "额度"])


def _has_numeric_fact(text: str) -> bool:
    patterns = [
        r"\d+(?:\.\d+)?\s*(天|岁|元|人民币|美元|wh|公斤|千克|ml|毫升|小时|分钟)",
        r"(不超过|不能超过|小于|大于|未满|满)\s*\d+",
    ]
    return any(re.search(p, text.lower()) for p in patterns)


def _is_duration_question(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in ["多久", "多长时间", "几小时", "几分钟", "提前", "几点", "何时", "什么时候"])


def _has_duration_fact(text: str) -> bool:
    t = text.lower()
    return bool(
        re.search(r"\d+(?:\.\d+)?\s*(小时|分钟|天|时|点)", t)
        or any(k in t for k in ["提前", "截至", "截止", "关闭", "时", "点"])
    )


def _normalize_for_matching(text: str) -> str:
    normalized = text.lower()
    synonym_map = {
        "现钞": "现金",
        "外币现钞": "外币现金",
        "进境": "入境",
        "出境": "离境",
        "折合": "兑换",
        "通关": "入境",
        "客户服务热线": "客服热线",
        "客户服务电话": "客服热线",
        "客服电话": "客服热线",
        "服务热线": "客服热线",
        "客服号码": "客服热线",
    }
    for src, dst in synonym_map.items():
        normalized = normalized.replace(src, dst)
    return normalized


def _extract_anchor_terms(question: str) -> set[str]:
    lowered = question.lower()
    for token in ["请问", "吗", "么", "呢", "如何", "怎么", "怎么办", "多少", "多久", "能否", "可以", "能够", "是否"]:
        lowered = lowered.replace(token, " ")
    base_terms = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]+", lowered))

    expanded: set[str] = set()
    for term in base_terms:
        expanded.add(term)
        if re.fullmatch(r"[\u4e00-\u9fff]{5,}", term):
            for n in (2, 3, 4):
                for i in range(0, len(term) - n + 1):
                    expanded.add(term[i : i + n])

    stop = {"规定", "要求", "办理", "旅客", "问题", "入境最多", "能携带", "多少现金"}
    return {t for t in expanded if t and t not in stop}


def _required_content_tokens(normalized_question: str) -> set[str]:
    raw_tokens = _extract_anchor_terms(normalized_question)
    stop = {
        "请问",
        "如何",
        "怎么",
        "怎么办",
        "需要",
        "可以",
        "能够",
        "是否",
        "多少",
        "多久",
        "一般",
        "问题",
        "规定",
        "要求",
        "流程",
    }
    return {t for t in raw_tokens if t not in stop and 2 <= len(t) <= 8}


def _required_intent_tokens(normalized_question: str) -> set[str]:
    rules = {
        "cash": {"现金", "现钞", "人民币", "美元", "货币"},
        "battery": {"充电宝", "锂电池", "电池", "wh", "瓦特小时"},
        "ticket": {"婴儿票", "儿童票", "成人票", "机票"},
    }

    intents: set[str] = set()
    if any(k in normalized_question for k in ["现金", "现钞", "人民币", "美元", "货币"]):
        intents.add("cash")
    has_wh_token = bool(re.search(r"\b\d+(?:\.\d+)?\s*wh\b|\bwh\b", normalized_question.lower()))
    if any(k in normalized_question for k in ["充电宝", "锂电池", "电池", "瓦特小时"]) or has_wh_token:
        intents.add("battery")
    if any(k in normalized_question for k in ["婴儿票", "儿童票", "成人票", "机票"]) or (
        "票" in normalized_question and any(k in normalized_question for k in ["岁", "周岁", "年龄"])
    ):
        intents.add("ticket")

    tokens: set[str] = set()
    for intent in intents:
        tokens.update(rules[intent])
    return tokens


def _char_bigrams(text: str) -> set[str]:
    cleaned = "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text.lower()))
    if len(cleaned) < 2:
        return {cleaned} if cleaned else set()
    return {cleaned[i : i + 2] for i in range(len(cleaned) - 1)}


def _bigram_similarity(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / union if union else 0.0
