from __future__ import annotations

from typing import Iterable

SELF_TEST_SEED_CASES: list[dict[str, str]] = [
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

    # 新增 25 题（展开后新增 100 题）
    # 海关（+5）
    {"topic": "海关", "question": "入境旅客现金达到什么标准要申报？", "expect": "answer"},
    {"topic": "海关", "question": "海关红通道适用于哪些情形？", "expect": "answer"},
    {"topic": "海关", "question": "绿色通道是不是就一定不用申报？", "expect": "answer"},
    {"topic": "海关", "question": "海关值班电话是多少？", "expect": "low-confidence"},
    {"topic": "海关", "question": "海关现场办理一般排队多久？", "expect": "low-confidence"},

    # 边防（+5）
    {"topic": "边防", "question": "外国籍港澳居民入境内地每次最多停留多少天？", "expect": "answer"},
    {"topic": "边防", "question": "外国人入境卡必须填写吗？", "expect": "answer"},
    {"topic": "边防", "question": "港澳居民来往内地应持哪些证件？", "expect": "answer"},
    {"topic": "边防", "question": "边检窗口具体在几号柜台？", "expect": "low-confidence"},
    {"topic": "边防", "question": "边检高峰时段平均等候时间多久？", "expect": "low-confidence"},

    # 出发（+5）
    {"topic": "出发", "question": "国内航班建议提前多长时间到航站楼？", "expect": "answer"},
    {"topic": "出发", "question": "国际航班出发通常提前多久到机场？", "expect": "answer"},
    {"topic": "出发", "question": "值机柜台关闭时间有统一标准吗？", "expect": "low-confidence"},
    {"topic": "出发", "question": "机场吸烟室在什么位置？", "expect": "low-confidence"},
    {"topic": "出发", "question": "登机口开放时间是固定的吗？", "expect": "low-confidence"},

    # 行李（+5）
    {"topic": "行李", "question": "100Wh以下充电宝可以随身带吗？", "expect": "answer"},
    {"topic": "行李", "question": "150Wh充电宝是否需要航司同意？", "expect": "answer"},
    {"topic": "行李", "question": "超过160Wh的充电宝还能带上飞机吗？", "expect": "answer"},
    {"topic": "行李", "question": "打火机能不能随身过安检？", "expect": "answer"},
    {"topic": "行李", "question": "托运行李超重收费标准是多少元每公斤？", "expect": "low-confidence"},

    # 行李-锂电池/充电宝专项（+8）
    {"topic": "行李", "question": "锂电池可以托运吗？", "expect": "answer"},
    {"topic": "行李", "question": "充电宝可以托运吗？", "expect": "answer"},
    {"topic": "行李", "question": "120Wh充电宝能随身携带吗？", "expect": "answer"},
    {"topic": "行李", "question": "160Wh充电宝可以带上飞机吗？", "expect": "answer"},
    {"topic": "行李", "question": "161Wh充电宝还能带吗？", "expect": "answer"},
    {"topic": "行李", "question": "20000mAh 3.7V充电宝可以带吗？", "expect": "answer"},
    {"topic": "行李", "question": "30000mAh 3.7V充电宝可以带吗？", "expect": "answer"},
    {"topic": "行李", "question": "没有标注Wh的充电宝能带上飞机吗？", "expect": "answer"},

    # 航司（+5）
    {"topic": "航司", "question": "南航官方客服热线电话是多少？", "expect": "answer"},
    {"topic": "航司", "question": "南航境外客服电话怎么联系？", "expect": "answer"},
    {"topic": "航司", "question": "春秋航空是否只有经济舱？", "expect": "answer"},
    {"topic": "航司", "question": "春秋航空提供免费机上餐食吗？", "expect": "answer"},
    {"topic": "航司", "question": "春秋航空最新客服电话号码是多少？", "expect": "low-confidence"},
]


QUICK_EVAL_QUESTIONS: list[str] = [
    "我的充电宝150Wh能带吗？",
    "我的充电宝170Wh能带吗？",
    "国内出发需要提前多久到达？",
    "国际出发一般提前多久到机场？",
    "港澳居民来往内地能停留多久？",
    "外国人入境是否需要填写入境卡？",
    "南航客服热线是多少？",
    "春秋航空有头等舱吗？",
    "1岁小孩应该买什么票？",
    "2岁小孩应该买什么票？",
    "机场和航司行李规定有什么区别？",
    "行李超重费用是多少？",
    "白云机场海关红色通道适用于哪些旅客？",
    "孕妇能否乘坐飞机？",
    "emirates自愿取消航班能退款吗？",
    "What is Emirates excess baggage policy?",
    "Can I get a refund if I voluntarily cancel my Emirates flight?",
    "国内出发值机柜台一般什么时候关闭？",
    "外国人入境卡在哪领取？",
    "入境最多能携带多少现金？",
    "锂电池可以托运吗？",
    "充电宝可以托运吗？",
    "20000mAh 3.7V充电宝可以带吗？",
]


DEFAULT_SELF_TEST_VARIANT_TEMPLATES: list[str] = [
    "{q}",
    "请问{q}",
    "依据现有规则，{q}",
    "{q}（请附依据）",
]


EXTENDED_SELF_TEST_VARIANT_TEMPLATES: list[str] = [
    *DEFAULT_SELF_TEST_VARIANT_TEMPLATES,
    "按机场现行口径，{q}",
    "请按规定答复：{q}",
    "用于回归测试：{q}",
    "给出可执行结论：{q}",
]


def expand_seed_cases(
    seed_cases: Iterable[dict[str, str]],
    variant_templates: Iterable[str],
) -> list[dict[str, str]]:
    expanded: list[dict[str, str]] = []
    for case in seed_cases:
        topic = case["topic"]
        question = case["question"]
        expect = case["expect"]
        for template in variant_templates:
            expanded.append(
                {
                    "topic": topic,
                    "question": template.format(q=question),
                    "expect": expect,
                }
            )
    return expanded


# API /self-test 默认题集（与历史行为一致：4种问法变体）
DEFAULT_SELF_TEST_CASES: list[dict[str, str]] = expand_seed_cases(
    SELF_TEST_SEED_CASES,
    DEFAULT_SELF_TEST_VARIANT_TEMPLATES,
)


# 已测试题全集（8种问法变体），用于回归抽样与批次管理
ALL_TESTED_CASES: list[dict[str, str]] = expand_seed_cases(
    SELF_TEST_SEED_CASES,
    EXTENDED_SELF_TEST_VARIANT_TEMPLATES,
)


# 两组不重复 200 题（共 400 题）
TESTED_QUESTION_BATCH_1_200: list[dict[str, str]] = ALL_TESTED_CASES[:200]
TESTED_QUESTION_BATCH_2_200: list[dict[str, str]] = ALL_TESTED_CASES[200:400]
