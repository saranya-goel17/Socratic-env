"""
Tests for SocraticEnv FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)


# ── Root & Health Tests ───────────────────────────────────

def test_root_returns_200():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "SocraticEnv"
    assert data["status"] == "running"


def test_ping_returns_healthy():
    r = client.get("/ping")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_metadata_endpoint():
    r = client.get("/metadata")
    assert r.status_code == 200
    data = r.json()
    assert "name" in data
    assert "description" in data
    assert data["name"] == "SocraticEnv"


def test_schema_endpoint():
    r = client.get("/schema")
    assert r.status_code == 200
    data = r.json()
    assert "action" in data
    assert "observation" in data
    assert "state" in data


def test_mcp_endpoint():
    r = client.post("/mcp", json={"method": "initialize", "id": 1})
    assert r.status_code == 200
    data = r.json()
    assert data["jsonrpc"] == "2.0"
    assert "result" in data


# ── Tasks Tests ───────────────────────────────────────────

def test_list_tasks_returns_all_five():
    r = client.get("/tasks")
    assert r.status_code == 200
    tasks = r.json()["tasks"]
    assert len(tasks) == 5
    task_ids = [t["id"] for t in tasks]
    assert "factual_recall" in task_ids
    assert "socratic_dialogue" in task_ids
    assert "misconception_trap" in task_ids
    assert "debate_mode" in task_ids
    assert "analogy_challenge" in task_ids


def test_tasks_have_required_fields():
    r = client.get("/tasks")
    tasks = r.json()["tasks"]
    for task in tasks:
        assert "id" in task
        assert "name" in task
        assert "difficulty" in task
        assert "description" in task


def test_tasks_difficulty_values():
    r = client.get("/tasks")
    tasks = r.json()["tasks"]
    valid_difficulties = ["easy", "medium", "hard"]
    for task in tasks:
        assert task["difficulty"] in valid_difficulties


# ── Reset Tests ───────────────────────────────────────────

def test_reset_factual_recall():
    r = client.post("/reset", json={"task_id": "factual_recall"})
    assert r.status_code == 200
    data = r.json()
    assert "observation" in data
    assert data["observation"]["task_id"] == "factual_recall"
    assert len(data["observation"]["question"]) > 0


def test_reset_socratic_dialogue():
    r = client.post("/reset", json={"task_id": "socratic_dialogue"})
    assert r.status_code == 200
    assert r.json()["observation"]["task_id"] == "socratic_dialogue"


def test_reset_misconception_trap():
    r = client.post("/reset", json={"task_id": "misconception_trap"})
    assert r.status_code == 200
    assert r.json()["observation"]["task_id"] == "misconception_trap"


def test_reset_debate_mode():
    r = client.post("/reset", json={"task_id": "debate_mode"})
    assert r.status_code == 200
    assert r.json()["observation"]["task_id"] == "debate_mode"


def test_reset_analogy_challenge():
    r = client.post("/reset", json={"task_id": "analogy_challenge"})
    assert r.status_code == 200
    assert r.json()["observation"]["task_id"] == "analogy_challenge"


def test_reset_invalid_task_returns_400():
    r = client.post("/reset", json={"task_id": "nonexistent_task"})
    assert r.status_code == 400


def test_reset_default_task():
    r = client.post("/reset", json={})
    assert r.status_code == 200


# ── Step Tests ────────────────────────────────────────────

def test_step_returns_reward_and_observation():
    client.post("/reset", json={"task_id": "factual_recall"})
    r = client.post("/step", json={"response": "Force equals mass times acceleration F=ma."})
    assert r.status_code == 200
    data = r.json()
    assert "reward" in data
    assert "observation" in data
    assert "done" in data
    assert "info" in data


def test_step_reward_in_valid_range():
    client.post("/reset", json={"task_id": "factual_recall"})
    r = client.post("/step", json={"response": "Force equals mass times acceleration."})
    score = r.json()["reward"]["score"]
    assert 0.0 <= score <= 1.0


def test_step_empty_response_returns_400():
    client.post("/reset", json={"task_id": "factual_recall"})
    r = client.post("/step", json={"response": ""})
    assert r.status_code == 400


def test_step_without_reset_returns_400():
    # Force done state by completing an episode
    client.post("/reset", json={"task_id": "factual_recall"})
    client.post("/step", json={"response": "Force and mass and acceleration F=ma."})
    client.post("/step", json={"response": "Doubling force doubles acceleration."})
    client.post("/step", json={"response": "No heavier objects do not accelerate faster."})
    # Now try to step again without reset
    r = client.post("/step", json={"response": "another response"})
    assert r.status_code == 400


def test_full_episode_all_tasks():
    """Each task completes a full episode without errors."""
    task_responses = {
        "factual_recall": [
            "Newton's Second Law states force equals mass times acceleration F=ma.",
            "Doubling force doubles acceleration since they are proportional.",
            "No that is incorrect heavier objects do not accelerate faster.",
        ],
        "debate_mode": [
            "Social media causes harm because research shows negative mental health effects.",
            "However social media provides benefits because it connects communities globally.",
            "I argue nuanced positions are more intellectually honest than absolute stances.",
            "Therefore I propose time limits and age verification as policy solutions.",
        ],
        "analogy_challenge": [
            "The internet is like a postal system where your computer sends letters to other computers.",
            "Clicking a link is like giving someone a new address to send their letter to.",
            "Slow websites are like traffic jams in the postal system with too many letters at once.",
        ],
    }

    for task_id, responses in task_responses.items():
        client.post("/reset", json={"task_id": task_id})
        for resp in responses:
            r = client.post("/step", json={"response": resp})
            assert r.status_code == 200
            data = r.json()
            assert 0.0 <= data["reward"]["score"] <= 1.0


# ── State Tests ───────────────────────────────────────────

def test_state_endpoint():
    client.post("/reset", json={"task_id": "factual_recall"})
    r = client.get("/state")
    assert r.status_code == 200
    data = r.json()
    assert "task_id" in data
    assert "turn" in data
    assert "done" in data
    assert "history" in data
    assert "total_score" in data


def test_state_updates_after_step():
    client.post("/reset", json={"task_id": "factual_recall"})
    client.post("/step", json={"response": "Force equals mass times acceleration."})
    r = client.get("/state")
    assert r.json()["turn"] == 1


# ── Leaderboard Tests ─────────────────────────────────────

def test_leaderboard_get():
    r = client.get("/leaderboard")
    assert r.status_code == 200
    data = r.json()
    assert "entries" in data
    assert "total" in data


def test_leaderboard_post_entry():
    entry = {
        "model_name": "Test Model pytest",
        "factual_recall": 0.75,
        "socratic_dialogue": 0.68,
        "misconception_trap": 0.60,
        "overall": 0.677,
    }
    r = client.post("/leaderboard", json=entry)
    assert r.status_code == 200
    assert r.json()["success"] == True


def test_leaderboard_delete_entry():
    # Add then delete
    entry = {
        "model_name": "DeleteMe pytest",
        "factual_recall": 0.5,
        "socratic_dialogue": 0.5,
        "misconception_trap": 0.5,
        "overall": 0.5,
    }
    client.post("/leaderboard", json=entry)
    r = client.delete("/leaderboard/DeleteMe pytest")
    assert r.status_code == 200
    assert r.json()["success"] == True