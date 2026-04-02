from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass

from .config import Settings, get_settings
from .embeddings import EmbeddingProvider
from .ingest import ingest_path
from .prompts import SYSTEM_STYLE, build_user_prompt
from .rules import (
    RuleResult,
    build_factoid_answer as _rule_build_factoid_answer,
    build_numeric_fact_answer as _rule_build_numeric_fact_answer,
    build_pregnancy_answer as _rule_build_pregnancy_answer,
    build_refund_rule_answer,
    build_rule_based_answer as _rule_build_rule_based_answer,
    extract_best_fact_sentence as _rule_extract_best_fact_sentence,
    has_duration_fact as _rule_has_duration_fact,
    is_contact_question as _rule_is_contact_question,
    is_duration_question as _rule_is_duration_question,
    is_fee_question as _rule_is_fee_question,
    is_numeric_question as _rule_is_numeric_question,
    localize_answer_text as _rule_localize_answer_text,
)
from .schemas import AskResponse, Citation, IngestResponse
from .vector_store import ChromaStore, RetrievedChunk


@dataclass
class _AnswerBundle:
    answer: str
    note: str
    evidence_chunk_ids: list[str] | None = None


@dataclass
class _GroundingResult:
    evidence: list[RetrievedChunk]
    reason: str


@dataclass
class _SourcePolicy:
    required_scope: str | None = None
    required_carrier: str | None = None
    preferred_scope: str | None = None


class AirportRAGService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedding = EmbeddingProvider(
            backend=self.settings.embedding_backend,
            model_name=self.settings.embedding_model,
        )
        collection_name = _build_collection_name(
            base=self.settings.collection_name,
            backend=self.settings.embedding_backend,
            model_name=self.settings.embedding_model,
        )
        self.store = ChromaStore(
            persist_dir=self.settings.vector_dir,
            collection_name=collection_name,
        )
        self._index_rebuild_attempted = False

    def ingest(self, input_path: str) -> IngestResponse:
        chunks, files = ingest_path(input_path, self.store, self.embedding)
        return IngestResponse(indexed_chunks=chunks, processed_files=files)

    def ask(self, question: str, top_k: int | None = None) -> AskResponse:
        self._ensure_index_ready()
        top_k = top_k or self.settings.top_k
        answer_lang = _detect_question_language(question)
        source_policy = _build_source_policy(question)
        retrieval_k = max(top_k * 12, 80)
        if source_policy.required_scope == "airline":
            retrieval_k = max(top_k * 20, 200)
        elif source_policy.required_scope == "airport":
            retrieval_k = max(top_k * 12, 100)
        where_filter = _build_vector_where_filter(source_policy)
        expanded_question = _expand_cross_lingual_query(question)
        query_vec = self.embedding.embed_query(expanded_question)
        retrieved = self.store.query(query_vec, top_k=retrieval_k, where=where_filter)

        if not retrieved and self.store.count() <= 0:
            text = (
                "结论：当前知识库索引为空，无法提供可核验的准确回答。\n"
                "依据：检索结果为空，且向量库中无已入库文档分片。\n"
                "执行建议：请先通过 /ingest 或管理端上传文档并完成入库，然后再提问。\n"
                "风险提示：在无证据情况下给出确定性结论可能误导业务执行，需避免。"
            )
            return AskResponse(
                question=question,
                answer=_localize_answer_text(text, answer_lang),
                citations=[],
                confidence_note="index-empty",
            )

        retrieved = _rerank_retrieved(expanded_question, retrieved)
        retrieved = _filter_retrieved_by_relevance(expanded_question, retrieved, keep_top=retrieval_k)
        raw_retrieved = list(retrieved)
        focused = _focus_retrieved(expanded_question, retrieved)
        focused = _filter_retrieved_by_relevance(expanded_question, focused, keep_top=top_k)
        answer_bundle = self._generate_answer(question, focused, raw_retrieved=raw_retrieved)
        answer_bundle.answer = _localize_answer_text(answer_bundle.answer, answer_lang)
        citations = self._to_citations(focused)
        if answer_bundle.note == "low-confidence":
            citations = []
        if answer_bundle.evidence_chunk_ids:
            wanted = set(answer_bundle.evidence_chunk_ids)
            filtered_citations = [c for c in citations if c.chunk_id in wanted]
            if filtered_citations:
                citations = filtered_citations
            else:
                citations = self._to_citations([r for r in raw_retrieved if r.chunk_id in wanted])

        return AskResponse(
            question=question,
            answer=answer_bundle.answer,
            citations=citations,
            confidence_note=answer_bundle.note,
        )

    def _ensure_index_ready(self) -> None:
        if self.store.count() > 0:
            return
        if self._index_rebuild_attempted:
            return

        self._index_rebuild_attempted = True
        docs_path = Path(__file__).resolve().parents[2] / "data" / "documents"
        if not docs_path.exists():
            return
        try:
            ingest_path(str(docs_path), self.store, self.embedding)
        except Exception:
            # Keep ask path resilient; caller will receive an index-empty response.
            return

    def _to_citations(self, retrieved: list[RetrievedChunk]) -> list[Citation]:
        citations: list[Citation] = []
        for idx, item in enumerate(retrieved, start=1):
            citations.append(
                Citation(
                    index=idx,
                    source=item.source,
                    page=item.page,
                    chunk_id=item.chunk_id,
                    snippet=item.text[:180],
                )
            )
        return citations

    def _generate_answer(
        self,
        question: str,
        retrieved: list[RetrievedChunk],
        raw_retrieved: list[RetrievedChunk] | None = None,
    ) -> _AnswerBundle:
        candidates = raw_retrieved or retrieved

        refund_rule = build_refund_rule_answer(question, candidates)
        if refund_rule is not None:
            return _AnswerBundle(
                answer=refund_rule.answer,
                note=refund_rule.note,
                evidence_chunk_ids=refund_rule.evidence_chunk_ids,
            )

        grounding = _select_grounded_evidence(question, candidates)
        grounded = grounding.evidence

        if not grounded:
            if _detect_question_language(question) == "zh" and _question_mentions_specific_carrier(question):
                fallback_q = _build_english_fallback_query(question)
                fallback_grounding = _select_grounded_evidence(fallback_q, candidates)
                if fallback_grounding.evidence:
                    grounded = fallback_grounding.evidence

        if not grounded:
            rule_fallback = _build_rule_based_answer(question, candidates)
            if rule_fallback is not None:
                return rule_fallback

            factoid_fallback = _build_factoid_answer(question, candidates)
            if factoid_fallback is not None:
                return factoid_fallback

            preg_fallback = _build_pregnancy_answer(question, candidates)
            if preg_fallback is not None:
                return preg_fallback
            return _AnswerBundle(
                answer=(
                    "结论：当前知识库未检索到可直接回答该问题的业务证据。\n"
                    f"依据：{grounding.reason}\n"
                    "执行建议：请补充更直接条款（含明确对象、限制条件和数值阈值）后再问。\n"
                    "风险提示：在证据不足情况下直接执行可能导致运行偏差，需人工复核。"
                ),
                note="low-confidence",
            )

        precise = _build_rule_based_answer(question, grounded)
        if precise is not None:
            return precise

        factoid = _build_factoid_answer(question, grounded)
        if factoid is not None:
            return factoid

        if _requires_specific_fact(question):
            return _AnswerBundle(
                answer=(
                    "结论：当前知识库未检索到可直接回答该问题的业务证据。\n"
                    "依据：已检索到相关主题片段，但缺少与问题问法一致的明确事实（如数值/时间/地点/明确可否结论）。\n"
                    "执行建议：请补充包含该问题关键字段的原文条款后再问。\n"
                    "风险提示：在证据不完整时给出确定结论可能导致误导，需人工复核。"
                ),
                note="low-confidence",
            )

        if self.settings.openai_api_key:
            try:
                from langchain_core.output_parsers import StrOutputParser
                from langchain_core.prompts import ChatPromptTemplate
                from langchain_openai import ChatOpenAI

                evidences = [{"source": r.source, "page": r.page, "text": r.text} for r in grounded]
                user_prompt = build_user_prompt(question, evidences)

                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", SYSTEM_STYLE),
                        ("user", "{user_prompt}"),
                    ]
                )

                model = ChatOpenAI(
                    model=self.settings.openai_model,
                    api_key=self.settings.openai_api_key,
                    temperature=0.1,
                )
                chain = prompt | model | StrOutputParser()
                content = chain.invoke({"user_prompt": user_prompt})
                return _AnswerBundle(
                    answer=content,
                    note="model-generated",
                    evidence_chunk_ids=[r.chunk_id for r in grounded[:3]],
                )
            except Exception:
                pass

        lines = [
            "结论：根据已检索证据，建议按文档要求执行对应机场业务流程。",
            "依据：",
        ]
        for idx, item in enumerate(grounded[:3], start=1):
            lines.append(
                f"- [{idx}] {item.text[:120]}（来源：{item.source}，页码：{item.page}）"
            )

        lines.extend(
            [
                "执行建议：由运行指挥/现场责任岗对照原文逐条核验后执行，关键环节保留审计记录。",
                "风险提示：当前回答基于检索片段自动生成，若涉及安全红线或航班正常性，请人工复核。",
            ]
        )
        return _AnswerBundle(
            answer="\n".join(lines),
            note="retrieval-extractive",
            evidence_chunk_ids=[r.chunk_id for r in grounded[:3]],
        )


def _detect_question_language(question: str) -> str:
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", question or ""))
    letters = re.findall(r"[A-Za-z]", question or "")
    if letters and not has_cjk:
        return "en"
    if has_cjk:
        return "zh"
    return "zh"


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

    code = _resolve_carrier_code_from_question(q)
    if code:
        alias_map = _load_carrier_alias_map()
        aliases = [k for k, v in alias_map.items() if v == code]
        extra.extend(aliases[:10])

    return q if not extra else f"{q} {' '.join(extra)}"


def _build_english_fallback_query(question: str) -> str:
    expanded = _expand_cross_lingual_query(question)
    english_tokens = re.findall(r"[A-Za-z0-9\-]{2,}", expanded)
    if not english_tokens:
        return expanded
    return " ".join(english_tokens)


def _localize_answer_text(text: str, lang: str) -> str:
    return _rule_localize_answer_text(text, lang)


def _bundle_from_rule(result: RuleResult | None) -> _AnswerBundle | None:
    if result is None:
        return None
    return _AnswerBundle(
        answer=result.answer,
        note=result.note,
        evidence_chunk_ids=result.evidence_chunk_ids,
    )


def _select_grounded_evidence(question: str, retrieved: list[RetrievedChunk], keep_top: int = 5) -> _GroundingResult:
    if not retrieved:
        return _GroundingResult(evidence=[], reason="未检索到相关文档片段。")

    expanded_q = _expand_cross_lingual_query(question)
    q_norm = _normalize_for_matching(expanded_q)
    anchors = _extract_anchor_terms(q_norm)
    content_tokens = _required_content_tokens(q_norm)
    salient_terms, salient_long_terms = _salient_terms(question)
    q_bigrams = _char_bigrams(q_norm)
    need_numeric = _is_numeric_question(question)
    required_tokens = _required_intent_tokens(q_norm)
    cue_groups = _expected_answer_cue_groups(question)

    ranked: list[tuple[tuple[int, int, float, float], RetrievedChunk]] = []
    for item in retrieved:
        joined = _normalize_for_matching(f"{item.source} {item.text}")
        req_hits = sum(1 for tok in required_tokens if tok in joined)
        if required_tokens and req_hits == 0:
            continue
        content_hits = sum(1 for tok in content_tokens if tok in joined)
        if content_tokens and content_hits == 0:
            continue
        salient_hits = sum(1 for tok in salient_terms if tok in joined)
        salient_long_hits = sum(1 for tok in salient_long_terms if tok in joined)
        if (not required_tokens) and salient_long_terms and salient_long_hits == 0 and salient_hits < 2:
            continue
        if cue_groups and not _matches_cue_groups(joined, cue_groups):
            continue
        anchor_hits = sum(1 for t in anchors if t in joined.lower())
        sim = _bigram_similarity(q_bigrams, _char_bigrams(joined))
        has_number = _has_numeric_fact(joined)

        if anchor_hits == 0 and sim < 0.08:
            continue
        if need_numeric and not has_number:
            continue

        ranked.append(((req_hits, anchor_hits, sim, -item.distance), item))

    ranked.sort(key=lambda p: p[0], reverse=True)
    evidence = [item for _, item in ranked[:keep_top]]

    if not evidence:
        if need_numeric:
            return _GroundingResult(evidence=[], reason="未找到同时包含问题核心对象与明确数值阈值的条款。")
        return _GroundingResult(evidence=[], reason="未找到与问题核心对象一致的条款。")
    return _GroundingResult(evidence=evidence, reason="已找到可支撑回答的直接条款。")


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


def _is_numeric_question(question: str) -> bool:
    return _rule_is_numeric_question(question)


def _has_numeric_fact(text: str) -> bool:
    patterns = [
        r"\d+(?:\.\d+)?\s*(天|岁|元|人民币|美元|wh|公斤|千克|ml|毫升|小时|分钟)",
        r"(不超过|不能超过|小于|大于|未满|满)\s*\d+",
    ]
    return any(re.search(p, text.lower()) for p in patterns)


def _is_duration_question(question: str) -> bool:
    return _rule_is_duration_question(question)


def _has_duration_fact(text: str) -> bool:
    return _rule_has_duration_fact(text)


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


def _build_numeric_fact_answer(question: str, retrieved: list[RetrievedChunk]) -> _AnswerBundle | None:
    return _bundle_from_rule(_rule_build_numeric_fact_answer(question, retrieved))


def _build_rule_based_answer(question: str, retrieved: list[RetrievedChunk]) -> _AnswerBundle | None:
    return _bundle_from_rule(_rule_build_rule_based_answer(question, retrieved))


def _build_factoid_answer(question: str, retrieved: list[RetrievedChunk]) -> _AnswerBundle | None:
    return _bundle_from_rule(_rule_build_factoid_answer(question, retrieved))


def _extract_best_fact_sentence(question: str, retrieved: list[RetrievedChunk]) -> tuple[str, RetrievedChunk | None]:
    return _rule_extract_best_fact_sentence(question, retrieved)


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


def _rerank_retrieved(question: str, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if len(retrieved) <= 1:
        return retrieved

    q_tokens = _tokenize(question)

    def _score(item: RetrievedChunk) -> tuple[int, float]:
        overlap = len(q_tokens.intersection(_tokenize(item.text)))
        return (-overlap, item.distance)

    return sorted(retrieved, key=_score)


def _focus_retrieved(question: str, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
    focused: list[RetrievedChunk] = []
    for item in retrieved:
        span = _extract_relevant_span(question, item.text)
        focused.append(
            RetrievedChunk(
                chunk_id=item.chunk_id,
                text=span,
                source=item.source,
                page=item.page,
                distance=item.distance,
            )
        )
    return focused


def _relevance_score(question: str, text: str) -> int:
    q_terms = _query_terms(question)
    q_tokens = _tokenize(question)
    term_overlap = sum(1 for term in q_terms if term in text)
    token_overlap = len(q_tokens.intersection(_tokenize(text)))
    return term_overlap * 10 + token_overlap


def _relevance_features(question: str, text: str) -> tuple[int, int]:
    q_terms = _query_terms(question)
    q_tokens = _tokenize(question)
    term_overlap = sum(1 for term in q_terms if term in text)
    token_overlap = len(q_tokens.intersection(_tokenize(text)))
    return term_overlap, token_overlap


def _filter_retrieved_by_relevance(
    question: str,
    retrieved: list[RetrievedChunk],
    keep_top: int = 3,
) -> list[RetrievedChunk]:
    if not retrieved:
        return retrieved

    source_policy = _build_source_policy(question)

    scored = sorted(
        (
            (
                item,
                _relevance_score(question, item.text)
                + _source_preference_bonus(source_policy, item),
            )
            for item in retrieved
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )

    q_topics = _infer_topics(question)
    expanded_question = _expand_question_with_topic_alias(question, q_topics)
    filtered: list[RetrievedChunk] = []
    for item, score in scored:
        if not _source_policy_compatible(source_policy, item):
            continue
        if not _intent_compatible(question, item.text, item.source):
            continue
        if q_topics:
            item_topics = _infer_topics(f"{item.source} {item.text}")
            if not q_topics.intersection(item_topics):
                continue
        term_overlap, token_overlap = _relevance_features(expanded_question, item.text)
        if term_overlap > 0 or token_overlap >= 2:
            filtered.append(item)

    if filtered:
        return filtered[:keep_top]

    compatible = []
    for item, _ in scored:
        if not _source_policy_compatible(source_policy, item):
            continue
        if not _intent_compatible(question, item.text, item.source):
            continue
        if q_topics:
            item_topics = _infer_topics(f"{item.source} {item.text}")
            if not q_topics.intersection(item_topics):
                continue
        compatible.append(item)

    if compatible:
        return compatible[:1]

    if not q_topics:
        return []

    if q_topics:
        return []
    return [item for item, _ in scored[:1]]


def _intent_compatible(question: str, text: str, source: str = "") -> bool:
    """Hard gate for mutually exclusive intent terms (e.g., 国内 vs 国际)."""
    q_domestic = "国内" in question
    q_international = "国际" in question
    joined = f"{source} {text}"
    t_domestic = "国内" in joined
    t_international = "国际" in joined

    if q_domestic and not q_international and t_international and not t_domestic:
        return False
    if q_international and not q_domestic and t_domestic and not t_international:
        return False
    return True


def _infer_topics(text: str) -> set[str]:
    lowered = text.lower()
    rules = {
        "departure": ["出发", "到达", "值机", "登机", "航站楼", "起飞", "departure", "arrival", "check-in", "boarding", "terminal"],
        "customs": ["海关", "申报", "红通道", "绿通道", "进境", "出境", "征税", "customs", "declare", "declaration"],
        "baggage": ["行李", "托运", "超重", "尺寸", "重量", "手提", "baggage", "luggage", "checked", "carry-on", "excess"],
        "battery": ["充电宝", "锂电池", "电池", "wh", "瓦特小时", "毫安", "额定能量", "battery", "power bank", "lithium"],
        "border": ["边防", "边检", "出入境", "港澳", "台湾", "通行证", "外国籍", "停留", "immigration", "border", "visa", "entry"],
    }
    topics: set[str] = set()
    for topic, keywords in rules.items():
        if any(keyword in lowered for keyword in keywords):
            topics.add(topic)
    return topics


def _expand_question_with_topic_alias(question: str, topics: set[str]) -> str:
    alias_map = {
        "battery": "充电宝 锂电池 额定能量 wh 毫安",
        "customs": "海关 申报 红通道 绿通道",
        "baggage": "行李 托运 手提 尺寸 重量",
        "departure": "出发 到达 值机 登机 航站楼",
        "border": "边防 边检 出入境 港澳 台湾 通行证 外国籍 停留",
    }
    parts = [question]
    for topic in topics:
        alias = alias_map.get(topic)
        if alias:
            parts.append(alias)
    return " ".join(parts)


def _build_collection_name(base: str, backend: str, model_name: str) -> str:
    signature = f"{backend}|{model_name}"
    suffix = hashlib.md5(signature.encode("utf-8")).hexdigest()[:10]
    return f"{base}_{suffix}"


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
    return _rule_is_fee_question(question)


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
    return _rule_is_contact_question(question)


def _build_pregnancy_answer(question: str, retrieved: list[RetrievedChunk]) -> _AnswerBundle | None:
    return _bundle_from_rule(_rule_build_pregnancy_answer(question, retrieved))


def _requires_specific_fact(question: str) -> bool:
    q = question.lower()
    if _is_contact_question(question):
        return True
    if _is_numeric_question(question):
        return True
    if _is_fee_question(question):
        return True
    if any(k in q for k in ["哪里", "在哪", "何处", "什么时候", "几点", "多久", "关闭", "截止", "领取", "上限", "阈值"]):
        return True
    if re.search(r"(能|可以|是否|可否).{0,6}吗", q):
        return True
    return False


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


def _build_source_policy(question: str) -> _SourcePolicy:
    q = question.lower()

    compare_keywords = [
        "区别",
        "不同",
        "一样",
        "相同",
        "近似",
        "对比",
        "分别",
        "各航司",
        "各航空公司",
        "difference",
        "compare",
        "similar",
    ]
    is_compare = any(k in q for k in compare_keywords)

    code = _resolve_carrier_code_from_question(question)
    if code:
        return _SourcePolicy(required_scope="airline", required_carrier=code, preferred_scope="airline")

    airport_keywords = [
        "白云机场",
        "机场规定",
        "机场",
        "航站楼",
        "海关",
        "边检",
        "边防",
        "入境卡",
        "外国人入境",
        "港澳居民",
        "airport",
        "terminal",
        "customs",
        "immigration",
        "border control",
    ]
    airline_keywords = [
        "航司",
        "航空公司",
        "承运人",
        "机票",
        "婴儿票",
        "儿童票",
        "退改签",
        "行李额",
        "客服",
        "热线",
        "客服电话",
        "孕妇",
        "妊娠",
        "产后",
        "分娩",
        "轮椅",
        "特殊旅客",
        "airline",
        "baggage",
        "refund",
        "rebook",
        "ticket",
        "customer service",
        "hotline",
    ]

    airport_hit = any(k in q for k in airport_keywords)
    airline_hit = any(k in q for k in airline_keywords)

    if is_compare and (airport_hit or airline_hit):
        return _SourcePolicy(preferred_scope=None)

    if airport_hit and not airline_hit:
        return _SourcePolicy(required_scope="airport", preferred_scope="airport")
    if airline_hit and not airport_hit:
        return _SourcePolicy(required_scope="airline", preferred_scope="airline")
    if airport_hit:
        return _SourcePolicy(preferred_scope="airport")
    if airline_hit:
        return _SourcePolicy(preferred_scope="airline")
    return _SourcePolicy(preferred_scope="airport")


def _build_vector_where_filter(policy: _SourcePolicy) -> dict | None:
    if policy.required_scope == "airline" and policy.required_carrier:
        return {"$and": [{"doc_scope": "airline"}, {"carrier": policy.required_carrier}]}
    if policy.required_scope:
        return {"doc_scope": policy.required_scope}
    return None


def _source_policy_compatible(policy: _SourcePolicy, item: RetrievedChunk) -> bool:
    scope, carrier = _source_scope_and_carrier(item)

    if policy.required_scope and scope != policy.required_scope:
        if scope != "unknown":
            return False
    if policy.required_scope and scope == "unknown":
        # tolerate legacy/unknown metadata in compatibility stage,
        # while vector pre-filter handles strict scope when metadata exists.
        pass
    if policy.required_carrier and carrier != policy.required_carrier:
        return False
    return True


def _source_preference_bonus(policy: _SourcePolicy, item: RetrievedChunk) -> int:
    scope, _ = _source_scope_and_carrier(item)
    if not policy.preferred_scope:
        return 0
    if scope == policy.preferred_scope:
        return 12
    if scope == "unknown":
        return 0
    return -4


def _question_mentions_specific_carrier(question: str) -> bool:
    return _resolve_carrier_code_from_question(question) is not None


def _resolve_carrier_code_from_question(question: str) -> str | None:
    alias_map = _load_carrier_alias_map()
    normalized_question = _normalize_carrier_key(question)

    for alias in sorted(alias_map.keys(), key=len, reverse=True):
        if alias and alias in normalized_question:
            return alias_map[alias]

    code_match = re.search(r"(?<![A-Za-z0-9])([A-Za-z0-9]{2})(?![A-Za-z0-9])", question)
    if code_match:
        code = code_match.group(1).upper()
        if code != "WH":
            return code
    return None


@lru_cache(maxsize=1)
def _load_carrier_alias_map() -> dict[str, str]:
    aliases: dict[str, str] = {
        "南航": "CZ",
        "中国南方航空": "CZ",
        "南方航空": "CZ",
        "东航": "MU",
        "国航": "CA",
        "春秋": "9C",
        "春秋航空": "9C",
        "阿联酋": "EK",
        "阿联酋航空": "EK",
        "emirates": "EK",
    }

    doc_path = Path(__file__).resolve().parents[2] / "data" / "documents" / "airport" / "航司代码"
    if not doc_path.exists():
        return {_normalize_carrier_key(k): v for k, v in aliases.items()}

    try:
        text = doc_path.read_text(encoding="utf-8")
    except Exception:
        return {_normalize_carrier_key(k): v for k, v in aliases.items()}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|") if c.strip()]
        if len(cells) < 2:
            continue

        code_idx = -1
        code_val = ""
        for i, cell in enumerate(cells):
            candidate = cell.strip().upper()
            if re.fullmatch(r"[A-Z0-9]{2}", candidate) and candidate != "WH":
                code_idx = i
                code_val = candidate
                break
        if code_idx < 0:
            continue

        for i, cell in enumerate(cells):
            if i == code_idx:
                continue
            for alias in _carrier_name_aliases(cell):
                aliases[alias] = code_val

    return {_normalize_carrier_key(k): v for k, v in aliases.items()}


def _carrier_name_aliases(company_name: str) -> set[str]:
    raw = company_name.strip()
    if not raw:
        return set()

    variants = {raw}
    variants.add(raw.lower())
    for src in ["股份有限公司", "有限责任公司", "航空公司", "航空", "公司"]:
        if src in raw:
            variants.add(raw.replace(src, ""))
    if raw.startswith("中国") and len(raw) > 2:
        variants.add(raw[2:])

    normalized_variants = {_normalize_carrier_key(v) for v in variants if v.strip()}
    expanded = set(normalized_variants)

    alias_hint = {
        "南方": "南航",
        "东方": "东航",
        "国际": "国航",
        "海南": "海航",
        "深圳": "深航",
        "厦门": "厦航",
        "山东": "山航",
        "春秋": "春秋",
        "吉祥": "吉祥",
        "emirates": "ek",
        "china southern": "cz",
        "china eastern": "mu",
        "air china": "ca",
        "spring airlines": "9c",
    }
    for key, alias in alias_hint.items():
        if any(key in name for name in normalized_variants):
            expanded.add(_normalize_carrier_key(alias))

    return {v for v in expanded if v}


def _normalize_carrier_key(text: str) -> str:
    normalized = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text or "")
    return normalized.lower()
