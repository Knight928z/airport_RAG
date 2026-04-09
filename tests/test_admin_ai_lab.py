from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from airport_rag import api as api_module


def test_admin_reranker_preview_returns_ranked_scores() -> None:
    client = TestClient(api_module.app)

    resp = client.post(
        "/admin/reranker/preview",
        json={
            "question": "国际到达是什么？",
            "candidates": [
                "国际到达流程请按指引前往到达层。",
                "托运行李超重请按航司规定处理。",
            ],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert "results" in body
    assert len(body["results"]) == 2
    assert body["results"][0]["score"] >= body["results"][1]["score"]


def test_admin_lora_train_accepts_job_and_returns_job_id(tmp_path: Path, monkeypatch) -> None:
    train_file = tmp_path / "train.jsonl"
    train_file.write_text('{"instruction":"问","output":"答"}\n', encoding="utf-8")

    monkeypatch.setattr(api_module, "LORA_ROOT", (tmp_path / "lora").resolve())

    monkeypatch.setattr(
        api_module,
        "start_lora_job",
        lambda cfg: {
            "job_id": "job-test-1",
            "status": "queued",
            "config": cfg,
            "created_at": "2026-04-09T12:00:00",
            "updated_at": "2026-04-09T12:00:00",
            "message": "queued",
            "output_dir": cfg["output_dir"],
        },
    )

    client = TestClient(api_module.app)
    resp = client.post(
        "/admin/lora/train",
        json={
            "train_file": str(train_file),
            "output_subdir": "airport-lora-test",
            "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.0002,
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "accepted"
    assert body["job"]["job_id"] == "job-test-1"
    assert body["job"]["config"]["train_file"] == str(train_file.resolve())


def test_admin_ai_lab_options_contains_presets_and_current_reranker(monkeypatch) -> None:
    monkeypatch.setattr(api_module.service.settings, "reranker_backend", "cross_encoder")
    monkeypatch.setattr(api_module.service.settings, "reranker_model", "my/custom-reranker")

    client = TestClient(api_module.app)
    resp = client.get("/admin/ai-lab/options")

    assert resp.status_code == 200
    body = resp.json()
    assert body["reranker"]["current_backend"] == "cross_encoder"
    assert body["reranker"]["current_model"] == "my/custom-reranker"
    assert "cross_encoder" in body["reranker"]["backend_options"]
    assert "heuristic" in body["reranker"]["backend_options"]
    assert "my/custom-reranker" in body["reranker"]["model_options"]
    assert "Qwen/Qwen2.5-0.5B-Instruct" in body["lora"]["base_model_options"]
