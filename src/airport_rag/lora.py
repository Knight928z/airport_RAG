from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class LoRAJob:
    job_id: str
    status: str
    created_at: str
    updated_at: str
    config: dict[str, Any]
    message: str = ""
    output_dir: str | None = None


_JOBS: dict[str, LoRAJob] = {}
_LOCK = threading.Lock()


def list_lora_jobs() -> list[dict[str, Any]]:
    with _LOCK:
        rows = list(_JOBS.values())
    rows.sort(key=lambda r: r.created_at, reverse=True)
    return [_job_to_dict(r) for r in rows]


def get_lora_job(job_id: str) -> dict[str, Any] | None:
    with _LOCK:
        row = _JOBS.get(job_id)
    return _job_to_dict(row) if row else None


def start_lora_job(config: dict[str, Any]) -> dict[str, Any]:
    now = _ts()
    job = LoRAJob(
        job_id=str(uuid4()),
        status="queued",
        created_at=now,
        updated_at=now,
        config=config,
    )
    with _LOCK:
        _JOBS[job.job_id] = job

    t = threading.Thread(target=_run_lora_job, args=(job.job_id,), daemon=True)
    t.start()
    return _job_to_dict(job)


def _run_lora_job(job_id: str) -> None:
    _update_job(job_id, status="running", message="开始加载训练数据")
    job = _snapshot(job_id)
    if not job:
        return

    try:
        cfg = job.config
        train_file = Path(str(cfg["train_file"]))
        out_dir = Path(str(cfg["output_dir"]))
        out_dir.mkdir(parents=True, exist_ok=True)

        examples = _load_examples(train_file)
        if not examples:
            raise ValueError("训练数据为空，请提供至少1条样本")

        _update_job(job_id, message=f"加载样本完成，共 {len(examples)} 条，开始构建 LoRA 训练")

        # Lazy import to avoid runtime dependency hard-fail when not using LoRA
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        base_model = str(cfg["base_model"])
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(base_model)

        target_modules = cfg.get("target_modules") or ["q_proj", "v_proj"]
        peft_cfg = LoraConfig(
            r=int(cfg.get("lora_r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 16)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)

        def _fmt(x: dict[str, Any]) -> str:
            ins = (x.get("instruction") or x.get("question") or "").strip()
            inp = (x.get("input") or "").strip()
            out = (x.get("output") or x.get("answer") or "").strip()
            if inp:
                return f"### 指令\n{ins}\n\n### 输入\n{inp}\n\n### 输出\n{out}"
            return f"### 指令\n{ins}\n\n### 输出\n{out}"

        rows = [{"text": _fmt(x)} for x in examples if _fmt(x).strip()]
        if not rows:
            raise ValueError("训练样本字段不完整，请提供 instruction/question 与 output/answer")

        ds = Dataset.from_list(rows)

        max_len = int(cfg.get("max_length", 512))

        def _tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
            return tokenizer(batch["text"], truncation=True, max_length=max_len)

        tokenized = ds.map(_tokenize, batched=True, remove_columns=["text"])

        args = TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=float(cfg.get("epochs", 1.0)),
            per_device_train_batch_size=int(cfg.get("batch_size", 2)),
            learning_rate=float(cfg.get("learning_rate", 2e-4)),
            logging_steps=int(cfg.get("logging_steps", 10)),
            save_strategy="epoch",
            report_to=[],
            fp16=False,
        )

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized,
            data_collator=collator,
        )

        _update_job(job_id, message="开始训练（LoRA）")
        trainer.train()

        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        _write_manifest(out_dir, cfg, len(rows))

        _update_job(
            job_id,
            status="succeeded",
            message=f"训练完成，权重输出到 {out_dir}",
            output_dir=str(out_dir),
        )
    except Exception as exc:
        _update_job(job_id, status="failed", message=f"训练失败: {exc}")


def _load_examples(train_file: Path) -> list[dict[str, Any]]:
    if not train_file.exists() or not train_file.is_file():
        raise FileNotFoundError(f"训练文件不存在: {train_file}")

    if train_file.suffix.lower() == ".json":
        data = json.loads(train_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        raise ValueError("JSON 文件需为对象数组")

    rows: list[dict[str, Any]] = []
    for line in train_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return [x for x in rows if isinstance(x, dict)]


def _write_manifest(output_dir: Path, cfg: dict[str, Any], sample_count: int) -> None:
    payload = {
        "created_at": _ts(),
        "sample_count": sample_count,
        "config": cfg,
    }
    (output_dir / "lora_train_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _snapshot(job_id: str) -> LoRAJob | None:
    with _LOCK:
        return _JOBS.get(job_id)


def _update_job(job_id: str, *, status: str | None = None, message: str | None = None, output_dir: str | None = None) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if not job:
            return
        if status is not None:
            job.status = status
        if message is not None:
            job.message = message
        if output_dir is not None:
            job.output_dir = output_dir
        job.updated_at = _ts()


def _job_to_dict(job: LoRAJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "status": job.status,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "config": job.config,
        "message": job.message,
        "output_dir": job.output_dir,
    }


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")
