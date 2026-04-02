#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 \"<type(scope): subject>\" [optional_note]"
  exit 1
fi

MSG="$1"
NOTE="${2:-}"

if [[ ! "$MSG" =~ ^(feat|fix|refactor|perf|docs|test|chore)(\([a-z0-9._-]+\))?:\ .+ ]]; then
  echo "Invalid commit message format."
  echo "Expected: type(scope): subject"
  echo "Example : feat(api): add feedback logging endpoint"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

git add -A

if [[ -z "$(git status --short)" ]]; then
  echo "No changes to commit."
  exit 0
fi

TS="$(date '+%Y-%m-%d %H:%M:%S %z')"
FILES="$(git diff --cached --name-only)"
export MSG NOTE TS FILES

python3 - <<'PY'
from pathlib import Path
import os

log_path = Path("CHANGELOG_LOG.md")
if not log_path.exists():
    log_path.write_text("# Change Log\n\n> 记录代码改动内容与时间戳（按时间倒序追加）。\n", encoding="utf-8")

content = log_path.read_text(encoding="utf-8")
msg = os.environ["MSG"]
ts = os.environ["TS"]
note = os.environ.get("NOTE", "").strip()
files = [f for f in os.environ.get("FILES", "").splitlines() if f.strip()]

entry = [f"## {ts}", "", f"- 提交：`{msg}`"]
if files:
    entry.append("- 变更文件：")
    entry.extend([f"  - `{f}`" for f in files])
if note:
    entry.append(f"- 备注：{note}")
entry_text = "\n".join(entry) + "\n\n"

anchor = "> 记录代码改动内容与时间戳（按时间倒序追加）。"
if anchor in content:
    idx = content.index(anchor) + len(anchor)
    new_content = content[:idx] + "\n\n" + entry_text + content[idx:].lstrip("\n")
else:
    new_content = content.rstrip() + "\n\n" + entry_text

log_path.write_text(new_content, encoding="utf-8")
PY

git add CHANGELOG_LOG.md
git commit -m "$MSG"
git push origin main

echo "Committed and pushed to origin/main: $MSG"
