from __future__ import annotations

import argparse
import json

from .service import AirportRAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="白云机场知识库 RAG 助手 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="文档入库")
    ingest_p.add_argument("path", help="文档目录或文件")

    ask_p = sub.add_parser("ask", help="检索问答")
    ask_p.add_argument("question", help="问题")
    ask_p.add_argument("--top-k", type=int, default=None)

    args = parser.parse_args()
    svc = AirportRAGService()

    if args.command == "ingest":
        result = svc.ingest(args.path)
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
    elif args.command == "ask":
        result = svc.ask(args.question, args.top_k)
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
