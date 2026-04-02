SYSTEM_STYLE = """
你是白云机场知识库专业助手。必须遵循：
1) 只依据给定证据回答，不得臆测；
2) 输出结构固定为：结论、依据、执行建议、风险提示；
3) 依据段落必须包含引用标记，如 [1] [2]；
4) 涉及运行标准、安保、机务、地服、航站楼运行时，措辞要审慎、可执行、可审计；
5) 证据不足时明确写“需人工复核”。
""".strip()


def build_user_prompt(question: str, evidences: list[dict]) -> str:
    evidence_text = "\n\n".join(
        f"[{i}] 来源: {ev['source']} | 页码: {ev.get('page')}\n内容: {ev['text']}"
        for i, ev in enumerate(evidences, start=1)
    )
    return (
        f"问题：{question}\n\n"
        f"可用证据如下：\n{evidence_text}\n\n"
        "请给出符合机场专业规范的回答，并在依据中标注引用编号。"
    )
