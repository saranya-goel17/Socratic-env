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
    assert len(tasks) == 7
    task_ids = [t["id"] for t in tasks]
    assert "factual_recall" in task_ids
    assert "socratic_dialogue" in task_ids
    assert "misconception_trap" in task_ids
    assert "debate_mode" in task_ids
    assert "analogy_challenge" in task_ids
    assert "cot_misconception" in task_ids
    assert "dynamic_misconception" in task_ids


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
    assert "session_id" in data
    assert data["observation"]["task_id"] == "factual_recall"
    assert len(data["observation"]["question"]) > 0


def test_reset_socratic_dialogue():
    r = client.post("/reset", json={"task_id": "socratic_dialogue"})
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert data["observation"]["task_id"] == "socratic_dialogue"


def test_reset_misconception_trap():
    r = client.post("/reset", json={"task_id": "misconception_trap"})
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert data["observation"]["task_id"] == "misconception_trap"


def test_reset_debate_mode():
    r = client.post("/reset", json={"task_id": "debate_mode"})
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert data["observation"]["task_id"] == "debate_mode"


def test_reset_analogy_challenge():
    r = client.post("/reset", json={"task_id": "analogy_challenge"})
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert data["observation"]["task_id"] == "analogy_challenge"


def test_reset_invalid_task_returns_400():
    r = client.post("/reset", json={"task_id": "nonexistent_task"})
    assert r.status_code == 400


def test_reset_default_task():
    r = client.post("/reset", json={})
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data


# ── Step Tests ────────────────────────────────────────────

def test_step_returns_reward_and_observation():
    reset_data = client.post("/reset", json={"task_id": "factual_recall"}).json()
    session_id = reset_data["session_id"]
    r = client.post("/step", json={
        "response": "Force equals mass times acceleration F=ma, which means acceleration depends on the net force and the object's mass.",
        "session_id": session_id
    })
    assert r.status_code == 200
    data = r.json()
    assert "reward" in data
    assert "observation" in data
    assert "done" in data
    assert "info" in data


def test_step_reward_in_valid_range():
    reset_data = client.post("/reset", json={"task_id": "factual_recall"}).json()
    session_id = reset_data["session_id"]
    r = client.post("/step", json={
        "response": "Force equals mass times acceleration, which is the fundamental relationship between these quantities in classical mechanics.",
        "session_id": session_id
    })
    score = r.json()["reward"]["score"]
    assert 0.0 <= score <= 1.0


def test_step_empty_response_returns_400():
    reset_data = client.post("/reset", json={"task_id": "factual_recall"}).json()
    session_id = reset_data["session_id"]
    r = client.post("/step", json={"response": "", "session_id": session_id})
    assert r.status_code == 400


def test_step_invalid_session_returns_404():
    """Step with a non-existent session_id should return 404."""
    r = client.post("/step", json={
        "response": "Some response here.",
        "session_id": "nonexistent-session-id"
    })
    assert r.status_code == 404


def test_step_after_done_returns_404():
    """After episode completes, session is cleaned up — next step returns 404."""
    reset_data = client.post("/reset", json={"task_id": "factual_recall"}).json()
    session_id = reset_data["session_id"]
    # Complete all 3 turns of factual_recall
    client.post("/step", json={
        "response": "Force and mass and acceleration F=ma, which describes how objects respond to applied forces in physics.",
        "session_id": session_id
    })
    client.post("/step", json={
        "response": "Doubling force doubles acceleration, since the relationship is directly proportional according to Newton's law.",
        "session_id": session_id
    })
    client.post("/step", json={
        "response": "No, heavier objects do not accelerate faster. In fact, with the same force a heavier object accelerates less.",
        "session_id": session_id
    })
    # Session should be cleaned up now — next step returns 404
    r = client.post("/step", json={
        "response": "another response that should fail.",
        "session_id": session_id
    })
    assert r.status_code == 404


def test_full_episode_all_tasks():
    """Each task completes a full episode without errors."""
    task_responses = {
        "factual_recall": [
            "Newton's Second Law states force equals mass times acceleration F=ma, describing the relationship between net force and motion.",
            "Doubling force doubles acceleration since they are proportional, as demonstrated by the equation F equals ma.",
            "No that is incorrect, heavier objects do not accelerate faster. With same force applied, heavier objects accelerate less.",
        ],
        "debate_mode": [
            "Social media causes harm because research shows negative mental health effects, especially among younger users today.",
            "However, social media provides benefits because it connects communities globally and enables rapid information sharing.",
            "I argue nuanced positions are more intellectually honest than absolute stances, because evidence supports both sides.",
            "Therefore I propose time limits and age verification as policy solutions, supported by evidence from multiple studies.",
        ],
        "analogy_challenge": [
            "The internet is like a postal system where your computer sends letters to other computers, similar to how mail routes work.",
            "Clicking a link is like giving someone a new address to send their letter to, just as you redirect mail delivery.",
            "Slow websites are like traffic jams in the postal system, imagine too many letters at once overwhelming the system.",
        ],
        "cot_misconception": [
            "Darwin's theory states species evolve through natural selection over many generations of gradual change.",
            "<think>The tutor claims organisms intentionally evolve, but this is incorrect because evolution is driven by random mutations. Therefore I must disagree with this false claim.</think> No, evolution is not intentional. It happens through random mutation and natural selection with no goal.",
        ],
        "dynamic_misconception": [
            "Darwin's theory of evolution explains that species change over time because natural selection favors beneficial traits.",
            "No I disagree. Evolution is not purposeful. Changes happen through random mutation and natural selection simply favours helpful traits.",
            "Natural selection works like a filter. Random mutations occur and helpful ones become more common over many generations.",
        ],
    }

    for task_id, responses in task_responses.items():
        reset_data = client.post("/reset", json={"task_id": task_id}).json()
        session_id = reset_data["session_id"]
        for resp in responses:
            r = client.post("/step", json={"response": resp, "session_id": session_id})
            assert r.status_code == 200
            data = r.json()
            assert 0.0 <= data["reward"]["score"] <= 1.0


# ── State Tests ───────────────────────────────────────────

def test_state_endpoint():
    reset_data = client.post("/reset", json={"task_id": "factual_recall"}).json()
    session_id = reset_data["session_id"]
    r = client.get(f"/state?session_id={session_id}")
    assert r.status_code == 200
    data = r.json()
    assert "task_id" in data
    assert "turn" in data
    assert "done" in data
    assert "history" in data
    assert "total_score" in data


def test_state_updates_after_step():
    reset_data = client.post("/reset", json={"task_id": "factual_recall"}).json()
    session_id = reset_data["session_id"]
    client.post("/step", json={
        "response": "Force equals mass times acceleration, which is the core principle of classical Newtonian mechanics.",
        "session_id": session_id
    })
    r = client.get(f"/state?session_id={session_id}")
    assert r.json()["turn"] == 1


def test_state_invalid_session_returns_404():
    """State with a non-existent session_id should return 404."""
    r = client.get("/state?session_id=nonexistent-session-id")
    assert r.status_code == 404


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


# ── Session Isolation Tests ──────────────────────────────

def test_concurrent_sessions_isolated():
    """Two sessions running in parallel should not interfere."""
    reset1 = client.post("/reset", json={"task_id": "factual_recall"}).json()
    reset2 = client.post("/reset", json={"task_id": "socratic_dialogue"}).json()
    sid1 = reset1["session_id"]
    sid2 = reset2["session_id"]

    assert sid1 != sid2

    # Step session 1
    r1 = client.post("/step", json={
        "response": "Force equals mass times acceleration F=ma, this is the fundamental equation of classical mechanics.",
        "session_id": sid1
    })
    assert r1.status_code == 200

    # Step session 2
    r2 = client.post("/step", json={
        "response": "Consciousness means the subjective experience of awareness, including self-reflection and perception of reality.",
        "session_id": sid2
    })
    assert r2.status_code == 200

    # Verify states are independent
    state1 = client.get(f"/state?session_id={sid1}").json()
    state2 = client.get(f"/state?session_id={sid2}").json()
    assert state1["task_id"] == "factual_recall"
    assert state2["task_id"] == "socratic_dialogue"


def test_session_cleanup_on_done():
    """Completed sessions are removed from active_sessions."""
    from main import active_sessions
    reset_data = client.post("/reset", json={"task_id": "factual_recall"}).json()
    session_id = reset_data["session_id"]
    assert session_id in active_sessions

    # Complete the episode
    client.post("/step", json={
        "response": "Force and mass and acceleration F=ma, describing how objects move under the influence of applied forces.",
        "session_id": session_id
    })
    client.post("/step", json={
        "response": "Doubling force doubles acceleration, since acceleration is directly proportional to force in this equation.",
        "session_id": session_id
    })
    client.post("/step", json={
        "response": "No, heavier objects do not accelerate faster. With the same force, heavier objects have less acceleration.",
        "session_id": session_id
    })

    # Session should be cleaned up
    assert session_id not in active_sessions