from types import SimpleNamespace
import airport_rag.service as service_module
from airport_rag.config import get_settings

from airport_rag.service import (
    AirportRAGService,
    _build_source_policy,
    _build_vector_where_filter,
    _resolve_carrier_code_from_question,
    _localize_answer_text,
    _build_collection_name,
    _build_factoid_answer,
    _build_rule_based_answer,
    _build_numeric_fact_answer,
    _select_grounded_evidence,
    _extract_relevant_span,
    _filter_retrieved_by_relevance,
    _focus_retrieved,
    _intent_compatible,
    _rerank_retrieved,
)
from airport_rag.rules import build_refund_rule_answer, build_ticket_insurance_refund_answer
from airport_rag.vector_store import RetrievedChunk


def _service_for_generate_answer_tests() -> AirportRAGService:
    svc = AirportRAGService.__new__(AirportRAGService)
    svc.settings = SimpleNamespace(openai_api_key=None)
    return svc


class _AskFakeEmbedding:
    def embed_query(self, question: str):
        return [0.0, 0.0, 0.0]


class _AskFakeReranker:
    def rerank(self, question: str, retrieved: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return retrieved


class _AskFakeStore:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def count(self) -> int:
        return 1

    def query(self, query_embedding, top_k: int, where=None):
        return list(self._chunks)


def _service_for_ask_tests(chunks: list[RetrievedChunk]) -> AirportRAGService:
    svc = AirportRAGService.__new__(AirportRAGService)
    svc.settings = SimpleNamespace(top_k=3, openai_api_key=None)
    svc.embedding = _AskFakeEmbedding()
    svc.reranker = _AskFakeReranker()
    svc.store = _AskFakeStore(chunks)
    svc._index_rebuild_attempted = True
    return svc


def test_rerank_prioritizes_keyword_overlap() -> None:
    items = [
        RetrievedChunk(
            chunk_id="1",
            text="航站楼值机与安检流程说明",
            source="A",
            page=None,
            distance=0.10,
        ),
        RetrievedChunk(
            chunk_id="2",
            text="托运行李超重费用与尺寸限制说明",
            source="B",
            page=None,
            distance=0.30,
        ),
    ]

    ranked = _rerank_retrieved("托运行李超过限制怎么办", items)

    assert ranked[0].chunk_id == "2"


def test_source_policy_routes_airport_only_question() -> None:
    p = _build_source_policy("海关红色通道适用于哪些旅客？")
    assert p.required_scope == "airport"


def test_source_policy_routes_airline_only_question() -> None:
    p = _build_source_policy("航空公司行李额规则是什么？")
    assert p.required_scope == "airline"


def test_source_policy_routes_battery_question_to_airport() -> None:
    p = _build_source_policy("我的充电宝150Wh能带吗？")
    assert p.required_scope is None
    assert p.preferred_scope == "airport"


def test_source_policy_routes_departure_time_question_to_airport() -> None:
    p = _build_source_policy("国内出发需要提前多久到达？")
    assert p.required_scope == "airport"


def test_source_policy_compare_question_not_hard_limited() -> None:
    p = _build_source_policy("机场和航司行李规定有什么区别？")
    assert p.required_scope is None


def test_rule_based_comparison_answer_contains_both_scopes() -> None:
    items = [
        RetrievedChunk(
            chunk_id="ap-1",
            text="机场对锂电池安检执行统一安全阈值检查。",
            source="/data/documents/airport/民航旅客限制随身携带或托运物品目录",
            page=None,
            distance=0.2,
            doc_scope="airport",
            carrier="",
        ),
        RetrievedChunk(
            chunk_id="cz-1",
            text="南航对托运行李重量和尺寸另有承运细则。",
            source="/data/documents/CZ/南航行李规定.pdf",
            page=None,
            distance=0.2,
            doc_scope="airline",
            carrier="CZ",
        ),
    ]

    ans = _build_rule_based_answer("机场和航司行李规定有什么区别？", items)

    assert ans is not None
    assert "双重规则" in ans.answer
    assert "[机场规则]" in ans.answer
    assert "[航司规则]" in ans.answer


def test_vector_where_filter_for_required_carrier() -> None:
    p = _build_source_policy("南航退改签规则是什么？")
    where = _build_vector_where_filter(p)
    assert where == {"$and": [{"doc_scope": "airline"}, {"carrier": "CZ"}]}


def test_extract_relevant_span_returns_focused_sentence() -> None:
    text = (
        "旅客进入航站楼后应先完成值机并关注登机口信息。"
        "海关检查须知：请提前准备护照、签证及海关申报材料，配合查验；"
        "若携带应申报物品请走红色通道，否则走绿色通道。"
        "登机口信息请以广播与电子屏为准。"
    )

    span = _extract_relevant_span("海关检查需要准备什么", text, max_chars=80)

    assert "海关检查" in span
    assert "护照" in span
    assert len(span) <= 80


def test_focus_retrieved_uses_span_not_whole_chunk() -> None:
    item = RetrievedChunk(
        chunk_id="c1",
        text=(
            "第一段与问题无关。"
            "托运行李超过限制时需按航空公司标准补交费用，并核验尺寸。"
            "第三段也与问题无关。"
        ),
        source="doc",
        page=None,
        distance=0.2,
    )

    focused = _focus_retrieved("托运行李超过限制怎么办", [item])[0]

    assert "补交费用" in focused.text
    assert len(focused.text) <= 180


def test_filter_retrieved_by_relevance_removes_unrelated() -> None:
    items = [
        RetrievedChunk(
            chunk_id="a",
            text="海关检查请准备护照并按申报要求通关。",
            source="doc-a",
            page=None,
            distance=0.1,
        ),
        RetrievedChunk(
            chunk_id="b",
            text="托运行李尺寸限制以航空公司为准。",
            source="doc-b",
            page=None,
            distance=0.1,
        ),
    ]

    filtered = _filter_retrieved_by_relevance("海关检查需要准备什么", items)

    assert len(filtered) == 1
    assert filtered[0].chunk_id == "a"


def test_filter_retrieved_preserves_reranker_order_instead_of_resorting() -> None:
    items = [
        RetrievedChunk(
            chunk_id="first",
            text="国际到达流程请按指引前往到达层。",
            source="doc-first",
            page=None,
            distance=0.2,
        ),
        RetrievedChunk(
            chunk_id="second",
            text="国际到达流程到达流程到达流程，请按流程办理。",
            source="doc-second",
            page=None,
            distance=0.1,
        ),
    ]

    filtered = _filter_retrieved_by_relevance("国际到达流程是什么", items, keep_top=2)

    assert len(filtered) == 2
    assert [x.chunk_id for x in filtered] == ["first", "second"]


def test_intent_compatible_blocks_international_for_domestic_question() -> None:
    question = "国内出发需要提前多久到达？"

    assert _intent_compatible(question, "国内出发建议提前2小时到达航站楼。") is True
    assert _intent_compatible(question, "国际出发建议提前3小时到达航站楼。") is False
    assert _intent_compatible(question, "提前3小时到达航站楼。", "/data/documents/出发指南-国际出发") is False


def test_filter_retrieved_prefers_departure_topic_for_departure_question() -> None:
    items = [
        RetrievedChunk(
            chunk_id="d",
            text="国内出发建议提前2小时到达航站楼办理值机。",
            source="/data/documents/出发指南-国内出发",
            page=None,
            distance=0.2,
        ),
        RetrievedChunk(
            chunk_id="c",
            text="可选择无申报通道通过海关。",
            source="/data/documents/海关检查须知",
            page=None,
            distance=0.1,
        ),
    ]

    filtered = _filter_retrieved_by_relevance("国内出发需要提前多久到达？", items)

    assert len(filtered) == 1
    assert filtered[0].chunk_id == "d"


def test_filter_retrieved_excludes_realtime_archive_for_general_arrival_question() -> None:
    items = [
        RetrievedChunk(
            chunk_id="rt-1",
            text="航班 CZ325 到达时间 12:35，状态延误。",
            source="/data/documents/airport/实时航班/2026-04-08-CZ325.md",
            page=None,
            distance=0.1,
        ),
        RetrievedChunk(
            chunk_id="arr-1",
            text="国际到达是指境外航班落地后旅客进入到达流程。",
            source="/data/documents/airport/到达指南-国际到达",
            page=None,
            distance=0.2,
        ),
    ]

    filtered = _filter_retrieved_by_relevance("国际到达是什么？", items)

    assert filtered
    assert all("/实时航班/" not in x.source.replace("\\", "/") for x in filtered)
    assert filtered[0].chunk_id == "arr-1"


def test_filter_retrieved_keeps_realtime_archive_for_realtime_flight_question() -> None:
    items = [
        RetrievedChunk(
            chunk_id="rt-2",
            text="MU2456 当前状态：延误 20 分钟。",
            source="/data/documents/airport/实时航班/2026-04-08-MU2456.md",
            page=None,
            distance=0.1,
        ),
        RetrievedChunk(
            chunk_id="arr-2",
            text="国际到达流程请按指引前往到达层。",
            source="/data/documents/airport/到达指南-国际到达",
            page=None,
            distance=0.2,
        ),
    ]

    filtered = _filter_retrieved_by_relevance("MU2456现在延误了吗？", items)

    assert filtered
    assert any("/实时航班/" in x.source.replace("\\", "/") for x in filtered)


def test_filter_retrieved_returns_empty_for_unsupported_battery_question() -> None:
    items = [
        RetrievedChunk(
            chunk_id="x1",
            text="国内出发建议提前2小时到达航站楼办理值机。",
            source="/data/documents/出发指南-国内出发",
            page=None,
            distance=0.1,
        ),
        RetrievedChunk(
            chunk_id="x2",
            text="海关申报物品请根据红绿通道规则通关。",
            source="/data/documents/海关检查须知",
            page=None,
            distance=0.2,
        ),
    ]

    filtered = _filter_retrieved_by_relevance("我的充电宝150Wh能带吗？", items)

    assert filtered == []


def test_filter_retrieved_matches_battery_synonym_document() -> None:
    items = [
        RetrievedChunk(
            chunk_id="b1",
            text="锂电池额定能量不超过160Wh，经航空公司同意可携带。",
            source="/data/documents/民航旅客限制随身携带或托运物品目录",
            page=None,
            distance=0.2,
        ),
    ]

    filtered = _filter_retrieved_by_relevance("我的充电宝150Wh能带吗？", items)

    assert len(filtered) == 1
    assert filtered[0].chunk_id == "b1"


def test_build_collection_name_isolated_by_embedding_signature() -> None:
    a = _build_collection_name("airport_kb", "hashing", "BAAI/bge-small-zh-v1.5")
    b = _build_collection_name("airport_kb", "sentence_transformers", "intfloat/multilingual-e5-large")

    assert a != b
    assert a.startswith("airport_kb_")


def test_rule_based_answer_for_battery_wh_question() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="1",
            text="充电宝额定能量超过100Wh但不超过160Wh，经航空公司同意后方可携带。",
            source="/data/documents/民航旅客限制随身携带或托运物品目录",
            page=None,
            distance=0.1,
        )
    ]

    ans = _build_rule_based_answer("我的充电宝150Wh能带吗？", retrieved)

    assert ans is not None
    assert "有条件可以" in ans.answer


def test_rule_based_answer_for_battery_over_100_without_allowance() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="1",
            text="充电宝、锂电池随身携带时额定能量应小于或等于100Wh。",
            source="/data/documents/民航旅客限制随身携带或托运物品目录",
            page=None,
            distance=0.1,
        )
    ]

    ans = _build_rule_based_answer("我的充电宝150Wh能带吗？", retrieved)

    assert ans is not None
    assert "进一步确认" in ans.answer or "人工复核" in ans.answer


def test_battery_answer_evidence_avoids_unrelated_alcohol_clause() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="bat-mixed-1",
            text=(
                "酒精的体积百分含量大于24%、小于或等于70%时，每位旅客托运数量不超过5L。"
                "充电宝、锂电池禁止作为行李托运，随身携带时额定能量应小于或等于100Wh。"
            ),
            source="/data/documents/airport/民航旅客限制随身携带或托运物品目录",
            page=None,
            distance=0.1,
        )
    ]

    ans = _build_rule_based_answer("我的充电宝150Wh能带吗？", retrieved)

    assert ans is not None
    assert "充电宝" in ans.answer
    assert "酒精的体积百分含量" not in ans.answer


def test_rule_based_answer_for_battery_wh_without_battery_keyword() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="bat-plain-wh-1",
            text="锂电池额定能量超过100Wh但不超过160Wh，经航空公司同意后方可携带。",
            source="/data/documents/airport/民航旅客限制随身携带或托运物品目录",
            page=None,
            distance=0.1,
        )
    ]

    ans = _build_rule_based_answer("120wh能带上飞机吗", retrieved)

    assert ans is not None
    assert "有条件可以" in ans.answer


def test_rule_based_answer_for_battery_mah_without_voltage_uses_estimation() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="bat-mah-1",
            text="充电宝额定能量超过100Wh但不超过160Wh，经航空公司同意后方可携带。",
            source="/data/documents/airport/民航旅客限制随身携带或托运物品目录",
            page=None,
            distance=0.1,
        )
    ]

    ans = _build_rule_based_answer("30000mAh充电宝可以带吗", retrieved)

    assert ans is not None
    assert "3.7V估算" in ans.answer
    assert "需航空公司同意" in ans.answer


def test_ask_battery_question_without_evidence_returns_low_confidence_without_citations() -> None:
    svc = _service_for_ask_tests(
        [
            RetrievedChunk(
                chunk_id="unrelated-1",
                text="国内出发建议提前2小时到达航站楼办理值机。",
                source="/data/documents/airport/出发指南-国内出发",
                page=None,
                distance=0.1,
                doc_scope="airport",
                carrier="",
            )
        ]
    )

    resp = svc.ask("我的充电宝150Wh能带吗？")

    assert resp.confidence_note == "low-confidence"
    assert resp.citations == []
    assert "未检索到可直接回答该问题的充电宝业务证据" in resp.answer


def test_ask_top_k_changes_rule_based_battery_evidence_count() -> None:
    chunks = [
        RetrievedChunk(
            chunk_id="ap-bat-1",
            text="充电宝、锂电池禁止作为行李托运，随身携带时有以下限定条件。",
            source="/data/documents/airport/民航旅客限制随身携带或托运物品目录",
            page=None,
            distance=0.1,
            doc_scope="airport",
            carrier="",
        ),
        RetrievedChunk(
            chunk_id="9c-bat-1",
            text="经航空公司批准，每位旅客最多可携带2块额定能量大于100Wh但不超过160Wh的备用锂电池乘机。",
            source="/data/documents/9C/春秋航空锂电池规定.png.ocr.md",
            page=None,
            distance=0.2,
            doc_scope="airline",
            carrier="9C",
        ),
        RetrievedChunk(
            chunk_id="9c-bat-2",
            text="严禁携带额定能量超过160Wh的锂电池。",
            source="/data/documents/9C/春秋航空锂电池规定.png.ocr.md",
            page=None,
            distance=0.3,
            doc_scope="airline",
            carrier="9C",
        ),
    ]
    svc = _service_for_ask_tests(chunks)

    resp_top1 = svc.ask("春秋航空120Wh充电宝可以带吗？", top_k=1)
    resp_top3 = svc.ask("春秋航空120Wh充电宝可以带吗？", top_k=3)

    assert resp_top1.confidence_note == "rule-based"
    assert len(resp_top1.citations) == 1
    assert "- [1]" in resp_top1.answer
    assert "- [2]" not in resp_top1.answer

    assert resp_top3.confidence_note == "rule-based"
    assert len(resp_top3.citations) == 3
    assert "- [3]" in resp_top3.answer


def test_rule_based_answer_for_customs_hotline_is_low_confidence() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="ap-border-12367",
            text="如需边检业务咨询可拨打12367。",
            source="/data/documents/airport/边防检查须知",
            page=None,
            distance=0.1,
        )
    ]

    ans = _build_rule_based_answer("海关值班电话是多少？", retrieved)

    assert ans is not None
    assert ans.note == "low-confidence"


def test_rule_based_answer_for_customs_queue_duration_is_low_confidence() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="ap-depart-time-1",
            text="按机场显示大屏及标识指引牌指示，起飞前30~40分钟到相应的登机口候机/登机。",
            source="/data/documents/airport/出发指南-国内出发",
            page=None,
            distance=0.1,
        )
    ]

    ans = _build_rule_based_answer("海关现场办理一般排队多久？", retrieved)

    assert ans is not None
    assert ans.note == "low-confidence"


def test_rule_based_answer_for_business_class_baggage_allowance() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="mu-comp-1",
            text="如行李发生延误，公务舱旅客人民币400元，经济舱旅客人民币200元。",
            source="/data/documents/MU/客票补偿标准",
            page=None,
            distance=0.1,
        ),
        RetrievedChunk(
            chunk_id="ap-bag-1",
            text="经济舱旅客免费行李额为20公斤，公务舱旅客为30公斤，头等舱旅客为40公斤。",
            source="/data/documents/airport/托运行李规定",
            page=None,
            distance=0.2,
        ),
    ]

    ans = _build_rule_based_answer("公务舱旅客能携带多少行李？", retrieved)

    assert ans is not None
    assert "30公斤" in ans.answer
    assert "人民币400元" not in ans.answer


def test_rule_based_answer_for_disabled_veteran_ticket_benefit() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="ap-military-1",
            text="军残票票价按照同一航班成人普通全票价的50％购买。",
            source="/data/documents/airport/机票办理指南",
            page=None,
            distance=0.1,
        ),
        RetrievedChunk(
            chunk_id="ap-military-2",
            text="如果您持有《残疾军人证》，可以直接拨打航空公司的热线电话预定机票。",
            source="/data/documents/airport/机票办理指南",
            page=None,
            distance=0.2,
        ),
    ]

    ans = _build_rule_based_answer("有残疾军人证有什么优待吗？", retrieved)

    assert ans is not None
    assert "50" in ans.answer
    assert "残疾军人证" in ans.answer


def test_ticket_insurance_refund_rule_for_9c() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="9c-ins-1",
            text="机票退票险购买后不支持退保，其余保险退保规定如下：旅客办理退票时，春秋航空所销售的保险可同机票一起退款。",
            source="/data/documents/9C/旅客须知",
            page=None,
            distance=0.1,
            doc_scope="airline",
            carrier="9C",
        )
    ]

    ans = build_ticket_insurance_refund_answer("9C退票时保险能退吗？", retrieved)

    assert ans is not None
    assert "保险" in ans.answer
    assert "退款" in ans.answer


def test_generic_refund_rule_skips_insurance_refund_question() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="9c-ins-2",
            text="机票退票险购买后不支持退保，其余保险可同机票一起退款。",
            source="/data/documents/9C/旅客须知",
            page=None,
            distance=0.1,
        )
    ]

    ans = build_refund_rule_answer("9C退票时保险能退吗？", retrieved)

    assert ans is None


def test_rule_based_answer_for_9c_free_meal_policy() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="9c-meal-1",
            text="除部分航班的尊享飞产品外，春秋航空不提供免费的餐饮。航班上将有偿提供多种餐食、饮料。",
            source="/data/documents/9C/旅客须知",
            page=None,
            distance=0.1,
            doc_scope="airline",
            carrier="9C",
        )
    ]

    ans = _build_rule_based_answer("9C有免费餐食吗？", retrieved)

    assert ans is not None
    assert "不提供免费的餐饮" in ans.answer


def test_rule_based_answer_for_foreign_tour_group_entry_documents() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="ap-border-1",
            text="外国旅游团入境时，应向边检机关交验护照、团体旅游签证名单表原件和复印件。",
            source="/data/documents/airport/边防检查须知",
            page=None,
            distance=0.1,
            doc_scope="airport",
            carrier="",
        ),
        RetrievedChunk(
            chunk_id="ap-depart-1",
            text="外国旅客须填写《外国人离境卡》（中国人无需填写）。",
            source="/data/documents/airport/出发指南-国际出发",
            page=None,
            distance=0.2,
            doc_scope="airport",
            carrier="",
        ),
    ]

    ans = _build_rule_based_answer("外国旅游团入境需要带什么？", retrieved)

    assert ans is not None
    assert "交验护照" in ans.answer
    assert "团体旅游签证名单表" in ans.answer


def test_rule_based_answer_for_domestic_flight_liquid_policy() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="ap-liquid-1",
            text="旅客乘坐国内航班时，液态物品禁止随身携带（航空旅行途中自用的化妆品、牙膏及剃须膏除外）。",
            source="/data/documents/airport/民航旅客限制随身携带或托运物品目录",
            page=None,
            distance=0.1,
            doc_scope="airport",
            carrier="",
        )
    ]

    ans = _build_rule_based_answer("国内航班能携带液体吗？", retrieved)

    assert ans is not None
    assert "一般不能随身携带" in ans.answer


def test_rule_based_answer_for_checked_lithium_battery() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="ap-battery-checked-1",
            text="充电宝、锂电池禁止作为行李托运，随身携带时有以下限定条件。",
            source="/data/documents/airport/民航旅客限制随身携带或托运物品目录",
            page=None,
            distance=0.1,
            doc_scope="airport",
            carrier="",
        )
    ]

    ans = _build_rule_based_answer("可以托运锂电池吗？", retrieved)

    assert ans is not None
    assert "不能托运" in ans.answer or "禁止作为行李托运" in ans.answer


def test_rule_based_battery_answer_combines_airport_and_airline_evidence_for_9c() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="ap-battery-1",
            text="充电宝、锂电池禁止作为行李托运，随身携带时有以下限定条件。",
            source="/data/documents/airport/民航旅客限制随身携带或托运物品目录",
            page=None,
            distance=0.1,
            doc_scope="airport",
            carrier="",
        ),
        RetrievedChunk(
            chunk_id="9c-battery-1",
            text="经航空公司批准，每位旅客最多可携带2块额定能量大于100Wh但不超过160Wh的备用锂电池乘机。严禁携带额定能量超过160Wh的锂电池。",
            source="/data/documents/9C/春秋航空锂电池规定.png.ocr.md",
            page=None,
            distance=0.2,
            doc_scope="airline",
            carrier="9C",
        ),
    ]

    ans = _build_rule_based_answer("春秋航空120Wh充电宝可以带吗？", retrieved)

    assert ans is not None
    assert "[机场规则]" in ans.answer
    assert "[航司规则]" in ans.answer
    assert "民航旅客限制随身携带或托运物品目录" in ans.answer
    assert "春秋航空锂电池规定" in ans.answer
    assert set(ans.evidence_chunk_ids or []) == {"ap-battery-1", "9c-battery-1"}


def test_rule_based_answer_for_infant_ticket_age_question() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="1",
            text="以起飞日期为准，未满2周岁应当购买婴儿票。",
            source="/data/documents/机票办理指南",
            page=None,
            distance=0.1,
        ),
        RetrievedChunk(
            chunk_id="2",
            text="以起飞日期为准，满2周岁但未满12周岁应当购买儿童票。",
            source="/data/documents/机票办理指南",
            page=None,
            distance=0.1,
        ),
    ]

    ans = _build_rule_based_answer("1岁应该购买什么票？", retrieved)

    assert ans is not None
    assert "婴儿票" in ans.answer


def test_rule_based_answer_for_border_stay_duration() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="border-1",
            text="外国籍港澳居民来往内地，可持用《港澳居民来往内地通行证（非中国籍）》，入境后每次停留不能超过90天。",
            source="/data/documents/边防检查须知",
            page=None,
            distance=0.1,
        )
    ]

    ans = _build_rule_based_answer("外国籍港澳居民来往内地能停留多久？", retrieved)

    assert ans is not None
    assert "90天" in ans.answer


    def test_rule_based_answer_for_pregnancy_restrictions() -> None:
        retrieved = [
            RetrievedChunk(
                chunk_id="mu-preg-1",
                text="计有分娩并发症的孕妇旅客、有先兆性流产反应的孕妇旅客及产后不足7天的旅客，不宜乘机或需提供医学证明。",
                source="/data/documents/MU/中国东方航空股份有限公司旅客、行李运输条件.pdf",
                page=None,
                distance=0.1,
            )
        ]

        ans = _build_rule_based_answer("孕妇能否乘坐飞机？", retrieved)

        assert ans is not None
        assert ans.note == "rule-based"
        assert "孕妇旅客" in ans.answer
        assert "限制" in ans.answer or "不宜乘机" in ans.answer


    def test_rule_based_answer_for_pregnancy_without_carrier_returns_guidance() -> None:
        retrieved = [
            RetrievedChunk(
                chunk_id="unrelated-1",
                text="国际航班建议提前3小时到达航站楼。",
                source="/data/documents/airport/出发指南-国际出发",
                page=None,
                distance=0.1,
            )
        ]

        ans = _build_rule_based_answer("孕妇能否乘坐飞机？", retrieved)

        assert ans is not None
        assert ans.note == "rule-based"
        assert "具体承运航司" in ans.answer


def test_border_stay_duration_avoids_transit_10_day_rule() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="transit-10",
            text="55国人员从中国过境前往第三国，可免签入境并在规定区域停留活动不超过10天。",
            source="/data/documents/边防检查须知",
            page=None,
            distance=0.05,
        ),
        RetrievedChunk(
            chunk_id="hk-90",
            text="外国籍港澳居民来往内地，可持用《港澳居民来往内地通行证（非中国籍）》，入境后每次停留不能超过90天。",
            source="/data/documents/边防检查须知",
            page=None,
            distance=0.2,
        ),
    ]

    ans = _build_rule_based_answer("外国籍港澳居民来往内地能停留多久？", retrieved)

    assert ans is not None
    assert "90天" in ans.answer


def test_grounding_rejects_offtopic_numeric_question() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="c1",
            text="每位年满18周岁成年旅客最多携带2名儿童同舱位乘机。",
            source="/data/documents/机票办理指南",
            page=None,
            distance=0.1,
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="外国籍港澳居民来往内地，每次停留不能超过90天。",
            source="/data/documents/边防检查须知",
            page=None,
            distance=0.2,
        ),
    ]

    grounding = _select_grounded_evidence("入境最多能携带多少现金？", retrieved)

    assert grounding.evidence == []


    def test_factoid_answer_rejects_non_duration_sentence_for_duration_question() -> None:
        retrieved = [
            RetrievedChunk(
                chunk_id="x-duration-1",
                text="根据国际航协（IATA）的相关规定，持有短期签证的乘客乘坐国际航班时，需出示返程机票。",
                source="/data/documents/机票办理指南",
                page=None,
                distance=0.1,
            )
        ]

        ans = _build_factoid_answer("国际航班安检和边检一般要预留多长时间？", retrieved)

        assert ans is None


def test_grounding_accepts_cash_clause_with_synonym() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="cash-1",
            text="人民币现钞超过20000元，或外币现钞折合超过5000美元。",
            source="/data/documents/海关检查须知",
            page=None,
            distance=0.1,
        )
    ]

    grounding = _select_grounded_evidence("入境最多能携带多少现金？", retrieved)

    assert len(grounding.evidence) == 1
    assert grounding.evidence[0].chunk_id == "cash-1"


def test_generic_numeric_fact_answer_for_cash_question() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="cash-1",
            text="人民币现钞超过20000元，或外币现钞折合超过5000美元。",
            source="/data/documents/海关检查须知",
            page=None,
            distance=0.1,
        )
    ]

    ans = _build_numeric_fact_answer("入境最多能携带多少现金？", retrieved)

    assert ans is not None
    assert "20000元" in ans.answer
    assert "5000美元" in ans.answer


def test_factoid_answer_for_hk_resident_document_question() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="doc-1",
            text="港澳居民来往内地，应持用《港澳居民来往内地通行证》。",
            source="/data/documents/边防检查须知",
            page=None,
            distance=0.1,
        )
    ]

    ans = _build_factoid_answer("港澳居民来往内地应该持什么证件？", retrieved)

    assert ans is not None
    assert "港澳居民来往内地通行证" in ans.answer


def test_factoid_answer_for_foreigner_entry_card_question() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="card-1",
            text="外国人入境应填写《外国人入境卡》。持团体签证、持外国人永久居留身份证的可免填写。",
            source="/data/documents/边防检查须知",
            page=None,
            distance=0.1,
        )
    ]

    ans = _build_factoid_answer("外国人入境是否需要填写入境卡？", retrieved)

    assert ans is not None
    assert "入境卡" in ans.answer


def test_numeric_fact_answer_rejects_non_fee_sentence_for_fee_question() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="bag-size-1",
            text="每件行李体积不超过20x40x55厘米。",
            source="/data/documents/托运行李规定",
            page=None,
            distance=0.1,
        )
    ]

    ans = _build_numeric_fact_answer("行李超重费用是多少？", retrieved)

    assert ans is None


def test_factoid_answer_rejects_unrelated_when_question_is_time() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="ticket-1",
            text="国内航班一般每位头等舱旅客可携带两件手提行李。",
            source="/data/documents/托运行李规定",
            page=None,
            distance=0.1,
        )
    ]

    ans = _build_factoid_answer("国内出发值机柜台一般什么时候关闭？", retrieved)

    assert ans is None


def test_factoid_answer_rejects_unrelated_location_question() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="dep-1",
            text="起飞前30~40分钟到相应登机口候机。",
            source="/data/documents/出发指南-国内出发",
            page=None,
            distance=0.1,
        )
    ]

    ans = _build_factoid_answer("机场有吸烟区吗？", retrieved)

    assert ans is None


def test_factoid_answer_rejects_pickup_location_mismatch() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="card-2",
            text="持团体签证、持外国人永久居留身份证、已备案可使用快捷通道的外国人免填写《外国人入境卡》。",
            source="/data/documents/边防检查须知",
            page=None,
            distance=0.1,
        )
    ]

    ans = _build_factoid_answer("外国人入境卡在哪领取？", retrieved)

    assert ans is None


def test_filter_retrieved_prefers_airport_folder_for_airport_question() -> None:
    items = [
        RetrievedChunk(
            chunk_id="ap-1",
            text="海关设置申报通道与无申报通道，按携带物品情况选择。",
            source="/data/documents/airport/海关检查须知",
            page=None,
            distance=0.2,
        ),
        RetrievedChunk(
            chunk_id="cz-1",
            text="南航值机流程说明：请提前到达柜台办理手续。",
            source="/data/documents/CZ/南航行李规定.pdf",
            page=1,
            distance=0.1,
        ),
    ]

    filtered = _filter_retrieved_by_relevance("白云机场海关红色通道怎么走？", items)

    assert filtered
    assert filtered[0].chunk_id == "ap-1"


def test_filter_retrieved_restricts_to_cz_for_southern_airlines_question() -> None:
    items = [
        RetrievedChunk(
            chunk_id="ap-1",
            text="婴儿票按起飞日期未满2周岁执行。",
            source="/data/documents/airport/机票办理指南",
            page=None,
            distance=0.1,
        ),
        RetrievedChunk(
            chunk_id="cz-1",
            text="南航婴儿票按起飞日期计算年龄，未满2周岁适用婴儿票。",
            source="/data/documents/CZ/南航行李规定.pdf",
            page=1,
            distance=0.3,
        ),
        RetrievedChunk(
            chunk_id="mu-1",
            text="东航婴儿票按公司规则执行。",
            source="/data/documents/MU/东航客票规则.pdf",
            page=1,
            distance=0.2,
        ),
    ]

    filtered = _filter_retrieved_by_relevance("南航婴儿票怎么规定？", items)

    assert filtered
    assert all("/data/documents/CZ/" in item.source for item in filtered)


def test_filter_retrieved_prefers_airline_folder_for_airline_question() -> None:
    items = [
        RetrievedChunk(
            chunk_id="ap-1",
            text="机场托运行李流程：请先值机再托运。",
            source="/data/documents/airport/托运行李规定",
            page=None,
            distance=0.1,
        ),
        RetrievedChunk(
            chunk_id="cz-1",
            text="航空公司托运行李规则：超重行李按公司标准处理。",
            source="/data/documents/CZ/南航行李规定.pdf",
            page=2,
            distance=0.2,
        ),
    ]

    filtered = _filter_retrieved_by_relevance("航空公司托运行李规定是什么？", items)

    assert filtered
    assert filtered[0].chunk_id == "cz-1"


def test_filter_retrieved_restricts_to_9c_for_spring_airlines_question() -> None:
    items = [
        RetrievedChunk(
            chunk_id="cz-1",
            text="南航客服热线(境内95539，境外+86-4008695539)。",
            source="/data/documents/CZ/客票退改规则/票价降低时客票处理规则",
            page=None,
            distance=0.1,
            doc_scope="airline",
            carrier="CZ",
        ),
        RetrievedChunk(
            chunk_id="9c-1",
            text="春秋航空为全经济舱布局，不设头等舱、商务舱。",
            source="/data/documents/9C/旅客须知",
            page=None,
            distance=0.2,
            doc_scope="airline",
            carrier="9C",
        ),
    ]

    filtered = _filter_retrieved_by_relevance("春秋航空客舱布局是什么？", items)

    assert filtered
    assert all("/data/documents/9C/" in item.source for item in filtered)


def test_rule_based_contact_answer_extracts_cz_hotline_numbers() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="cz-contact-1",
            text="南航直属销售单位、客户服务热线(境内95539，境外+86-4008695539)及南航航空销售代理人。",
            source="/data/documents/CZ/客票退改规则/因病退改规则",
            page=None,
            distance=0.1,
            doc_scope="airline",
            carrier="CZ",
        )
    ]

    ans = _build_rule_based_answer("南航客服热线是多少？", retrieved)

    assert ans is not None
    assert "95539" in ans.answer
    assert "4008695539" in ans.answer


def test_rule_based_contact_answer_extracts_airport_hotline_numbers() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="airport-contact-1",
            text="广州白云机场客服热线为020-96158，可提供旅客咨询服务。",
            source="/data/documents/airport/机场服务指南",
            page=None,
            distance=0.1,
            doc_scope="airport",
            carrier="",
        )
    ]

    ans = _build_rule_based_answer("机场客服热线是多少？", retrieved)

    assert ans is not None
    assert "96158" in ans.answer


def test_contact_question_without_number_does_not_create_contact_rule_answer() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="cz-contact-2",
            text="可联系南航直属销售单位或客服热线办理业务。",
            source="/data/documents/CZ/客票退改规则/因病退改规则",
            page=None,
            distance=0.1,
            doc_scope="airline",
            carrier="CZ",
        )
    ]

    ans = _build_rule_based_answer("南航客服电话是多少？", retrieved)

    assert ans is None


def test_contact_factoid_requires_phone_number() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="contact-no-number",
            text="如您的机票需要改签，可直接拨打航空公司热线电话进行改签。",
            source="/data/documents/airport/机票办理指南",
            page=None,
            distance=0.1,
        )
    ]

    ans = _build_factoid_answer("航空公司客服热线是多少？", retrieved)

    assert ans is None


def test_numeric_fact_answer_rejects_irrelevant_time_clause() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="border-24h",
            text="在机场口岸停留不超过24小时且不离开口岸限定区域，可免办边检手续。",
            source="/data/documents/边防检查须知",
            page=None,
            distance=0.1,
        )
    ]

    ans = _build_numeric_fact_answer("边检人工通道平均排队多久？", retrieved)

    assert ans is None


def test_resolve_carrier_code_from_question_uses_airline_aliases() -> None:
    assert _resolve_carrier_code_from_question("南航客服热线是多少？") == "CZ"
    assert _resolve_carrier_code_from_question("春秋航空是否提供免费餐饮？") == "9C"


def test_resolve_carrier_code_from_question_supports_code_literal() -> None:
    assert _resolve_carrier_code_from_question("9C托运行李规则是什么？") == "9C"
    assert _resolve_carrier_code_from_question("CZ退改签如何办理？") == "CZ"


def test_resolve_carrier_code_from_question_supports_english_name() -> None:
    assert _resolve_carrier_code_from_question("What is Emirates baggage policy?") == "EK"


def test_localize_answer_text_for_english_question() -> None:
    zh = "结论：可联系的客服热线。\n依据：\n执行建议：请联系航司。\n风险提示：请以最新公告为准。"
    en = _localize_answer_text(zh, "en")
    assert "Conclusion:" in en
    assert "Evidence:" in en
    assert "Recommendation:" in en
    assert "Risk note:" in en


def test_refund_rule_handles_emirates_voluntary_cancel_in_chinese() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="ek-refund-1",
            text="You voluntarily cancel your flight; you are not entitled to a refund of your purchase.",
            source="/data/documents/EK/refund-policy",
            page=None,
            distance=0.1,
            doc_scope="airline",
            carrier="EK",
        )
    ]

    ans = build_refund_rule_answer("emirates自愿取消航班能退款吗？", retrieved)

    assert ans is not None
    assert "不能退款" in ans.answer
    assert "自愿取消航班" in ans.answer
    assert "ek-refund-1" in (ans.evidence_chunk_ids or [])


def test_refund_rule_handles_emirates_voluntary_cancel_in_english() -> None:
    retrieved = [
        RetrievedChunk(
            chunk_id="ek-refund-2",
            text="You voluntarily cancel your flight; you are not entitled to a refund of your purchase.",
            source="/data/documents/EK/refund-policy",
            page=None,
            distance=0.1,
            doc_scope="airline",
            carrier="EK",
        )
    ]

    ans = build_refund_rule_answer("If I voluntarily cancel my Emirates flight, can I get a refund?", retrieved)

    assert ans is not None
    assert "not refundable" in ans.answer
    assert "voluntary flight cancellation" in ans.answer.lower()
    assert "ek-refund-2" in (ans.evidence_chunk_ids or [])


def test_generate_answer_falls_back_to_rule_based_when_grounding_empty() -> None:
    svc = _service_for_generate_answer_tests()
    raw = [
        RetrievedChunk(
            chunk_id="bat-raw-1",
            text="超过100Wh但不超过160Wh，经航空公司同意后方可携带。",
            source="/data/documents/airport/民航旅客限制随身携带或托运物品目录",
            page=None,
            distance=0.1,
        )
    ]

    ans = svc._generate_answer("我的充电宝150Wh能带吗？", [], raw_retrieved=raw)

    assert ans.note == "rule-based"
    assert "有条件可以" in ans.answer
    assert "bat-raw-1" in (ans.evidence_chunk_ids or [])


def test_generate_answer_falls_back_to_factoid_when_grounding_empty() -> None:
    svc = _service_for_generate_answer_tests()
    raw = [
        RetrievedChunk(
            chunk_id="dep-raw-1",
            text="建议提前2小时到达航站楼办理值机。",
            source="/data/documents/airport/出发指南-国内出发",
            page=None,
            distance=0.1,
        )
    ]

    ans = svc._generate_answer("国内出发需要提前多久到达？", [], raw_retrieved=raw)

    assert ans.note in {"grounded-factoid", "rule-based"}
    assert "提前2小时" in ans.answer
    assert "dep-raw-1" in (ans.evidence_chunk_ids or [])


def test_generate_answer_uses_local_lora_when_specific_fact_required(monkeypatch) -> None:
    svc = _service_for_generate_answer_tests()
    svc.settings = SimpleNamespace(
        openai_api_key=None,
        generation_backend="local_lora",
        lora_base_model="Qwen/Qwen2.5-0.5B-Instruct",
        lora_adapter_path="/tmp/mock-adapter",
        lora_max_new_tokens=128,
    )
    retrieved = [
        RetrievedChunk(
            chunk_id="ground-1",
            text="海关现场排队时长受时段客流影响，需以现场显示为准。",
            source="/data/documents/airport/海关检查须知",
            page=None,
            distance=0.1,
        )
    ]

    monkeypatch.setattr(service_module, "_build_rule_based_answer", lambda question, chunks: None)
    monkeypatch.setattr(service_module, "_build_factoid_answer", lambda question, chunks: None)
    monkeypatch.setattr(service_module, "_requires_specific_fact", lambda question: True)
    monkeypatch.setattr(
        service_module,
        "_select_grounded_evidence",
        lambda question, chunks, keep_top=5: SimpleNamespace(evidence=chunks, reason="ok"),
    )
    monkeypatch.setattr(
        service_module,
        "_generate_with_local_lora",
        lambda question, grounded, settings: "结论：需以现场排队显示为准。\n依据：已结合检索证据。",
    )

    ans = svc._generate_answer("海关人工通道平均排队多久？", retrieved, raw_retrieved=retrieved)

    assert ans.note == "local-lora-generated"
    assert "需以现场排队显示为准" in ans.answer
    assert "ground-1" in (ans.evidence_chunk_ids or [])


def test_default_generation_backend_is_disabled_for_rag_first() -> None:
    settings = get_settings()

    assert settings.generation_backend in {"disabled", "off", "none"}
