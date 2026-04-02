from airport_rag.ingest import split_text


def test_split_text_basic() -> None:
    text = "A" * 1200
    chunks = split_text(text, chunk_size=500, overlap=100)
    assert len(chunks) == 3
    assert len(chunks[0]) == 500


def test_split_text_empty() -> None:
    assert split_text("   ") == []


def test_split_text_sentence_aware() -> None:
    text = "第一条：旅客到达后先值机。\n第二条：海关检查请准备护照与申报材料。\n第三条：按广播提示登机。"
    chunks = split_text(text, chunk_size=40, overlap=10)

    assert len(chunks) >= 2
    assert any("海关检查" in c for c in chunks)
