from __future__ import annotations

import argparse
import json
from typing import Iterable

from .eval_cases import ALL_TESTED_CASES, DEFAULT_SELF_TEST_CASES, TESTED_QUESTION_BATCH_1_200, TESTED_QUESTION_BATCH_2_200
from .service import AirportRAGService


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="白云机场知识库 RAG 助手 CLI",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m airport_rag.cli ingest ./data/documents\n"
            "  python -m airport_rag.cli ask \"国际到达是什么\" --show-citations\n"
            "  python -m airport_rag.cli self-test --batch batch2 --limit 200"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="文档入库")
    ingest_p.add_argument("path", nargs="?", default="./data/documents", help="文档目录或文件（默认 ./data/documents）")
    ingest_p.add_argument("--json", action="store_true", help="以 JSON 输出")

    ask_p = sub.add_parser("ask", help="检索问答")
    ask_p.add_argument("question", help="问题")
    ask_p.add_argument("--top-k", type=int, default=None, help="引用条数（默认读取配置）")
    ask_p.add_argument("--json", action="store_true", help="输出完整 JSON")
    ask_p.add_argument("--answer-only", action="store_true", help="仅输出答案正文")
    ask_p.add_argument("--show-citations", action="store_true", help="文本模式下附带引用")
    ask_p.add_argument("--show-meta", action="store_true", help="文本模式下附带置信状态与引用统计")

    test_p = sub.add_parser("self-test", help="本地回归自测（无需启动 API）")
    test_p.add_argument(
        "--batch",
        choices=["default", "batch1", "batch2", "all"],
        default="default",
        help="题集批次：default=接口默认题集；batch1/batch2=两组不重复200题；all=全部已测题",
    )
    test_p.add_argument("--limit", type=int, default=None, help="最多执行题数")
    test_p.add_argument("--show-failures", type=int, default=20, help="显示失败样例数（默认20）")
    test_p.add_argument("--json", action="store_true", help="以 JSON 输出结果")
    return parser


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _cases_for_batch(batch: str) -> list[dict[str, str]]:
    if batch == "batch1":
        return TESTED_QUESTION_BATCH_1_200
    if batch == "batch2":
        return TESTED_QUESTION_BATCH_2_200
    if batch == "all":
        return ALL_TESTED_CASES
    return DEFAULT_SELF_TEST_CASES


def _run_self_test(
    svc: AirportRAGService,
    cases: Iterable[dict[str, str]],
    show_failures: int,
) -> dict:
    total = 0
    passed = 0
    failed = 0
    errors = 0
    low_confidence = 0
    failed_rows: list[dict[str, str]] = []

    for case in cases:
        total += 1
        q = case["question"]
        expect = case["expect"]
        topic = case.get("topic", "未分类")
        try:
            r = svc.ask(q)
            actual = "low-confidence" if r.confidence_note in {"low-confidence", "index-empty"} else "answer"
            ok = actual == expect
            if ok:
                passed += 1
            else:
                failed += 1
                failed_rows.append(
                    {
                        "topic": topic,
                        "question": q,
                        "expect": expect,
                        "actual": actual,
                        "confidence": r.confidence_note,
                        "first_line": (r.answer or "").split("\n")[0],
                    }
                )
            if r.confidence_note == "low-confidence":
                low_confidence += 1
        except Exception as exc:
            failed += 1
            errors += 1
            failed_rows.append(
                {
                    "topic": topic,
                    "question": q,
                    "expect": expect,
                    "actual": "error",
                    "confidence": "error",
                    "first_line": str(exc),
                }
            )

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "errors": errors,
        "low_confidence": low_confidence,
        "failed_samples": failed_rows[: max(show_failures, 0)],
    }


def _print_ask_text(result, show_citations: bool, show_meta: bool) -> None:
    print(result.answer)
    if show_meta:
        print(f"\n[meta] confidence={result.confidence_note} citations={len(result.citations)}")
    if show_citations and result.citations:
        print("\n[citations]")
        for c in result.citations:
            page = f", p.{c.page}" if c.page is not None else ""
            print(f"- [{c.index}] {c.source}{page} | {c.snippet}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    svc = AirportRAGService()

    if args.command == "ingest":
        result = svc.ingest(args.path)
        payload = result.model_dump()
        if args.json:
            _print_json(payload)
        else:
            print(f"入库完成: indexed_chunks={payload['indexed_chunks']} processed_files={payload['processed_files']}")
    elif args.command == "ask":
        result = svc.ask(args.question, args.top_k)
        if args.json:
            _print_json(result.model_dump())
        elif args.answer_only:
            print(result.answer)
        else:
            _print_ask_text(result, show_citations=args.show_citations, show_meta=args.show_meta)
    elif args.command == "self-test":
        try:
            svc.ingest("./data/documents")
        except Exception:
            pass

        cases = _cases_for_batch(args.batch)
        if args.limit is not None and args.limit >= 0:
            cases = cases[: args.limit]

        summary = _run_self_test(svc, cases, show_failures=args.show_failures)
        summary["batch"] = args.batch

        if args.json:
            _print_json(summary)
        else:
            print(
                "self-test: "
                f"batch={args.batch} total={summary['total']} passed={summary['passed']} "
                f"failed={summary['failed']} pass_rate={summary['pass_rate']} "
                f"errors={summary['errors']} low_confidence={summary['low_confidence']}"
            )
            if summary["failed_samples"]:
                print("\nfailed samples:")
                for row in summary["failed_samples"]:
                    print(f"- [{row['topic']}] {row['question']}")
                    print(
                        f"  expect={row['expect']} actual={row['actual']} "
                        f"confidence={row['confidence']}"
                    )
                    print(f"  first_line={row['first_line']}")


if __name__ == "__main__":
    main()
