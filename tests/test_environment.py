"""
Tests for SocraticEnv core environment logic.
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import (
    SocraticEnvironment,
    Action,
    Observation,
    Reward,
    StepResult,
    StateInfo,
)


# ── Fixtures ──────────────────────────────────────────────

@pytest.fixture
def env():
    """Fresh environment for each test."""
    return SocraticEnvironment()


@pytest.fixture(autouse=True)
def mock_random_choice(monkeypatch):
    """Ensure random.choice always picks the first topic for deterministic testing."""
    monkeypatch.setattr("environment.random.choice", lambda seq: seq[0])


# ── Reset Tests ───────────────────────────────────────────

def test_reset_factual_recall(env):
    obs = env.reset("factual_recall")
    assert isinstance(obs, Observation)
    assert obs.task_id == "factual_recall"
    assert obs.turn == 0
    assert len(obs.question) > 0
    assert env.done == False
    assert env.max_turns == 3


def test_reset_socratic_dialogue(env):
    obs = env.reset("socratic_dialogue")
    assert isinstance(obs, Observation)
    assert obs.task_id == "socratic_dialogue"
    assert env.max_turns == 5
    assert env.done == False


def test_reset_misconception_trap(env):
    obs = env.reset("misconception_trap")
    assert isinstance(obs, Observation)
    assert obs.task_id == "misconception_trap"
    assert env.max_turns == 4
    assert env.done == False


def test_reset_debate_mode(env):
    obs = env.reset("debate_mode")
    assert isinstance(obs, Observation)
    assert obs.task_id == "debate_mode"
    assert env.max_turns == 4
    assert env.done == False


def test_reset_analogy_challenge(env):
    obs = env.reset("analogy_challenge")
    assert isinstance(obs, Observation)
    assert obs.task_id == "analogy_challenge"
    assert env.max_turns == 3
    assert env.done == False


def test_reset_invalid_task(env):
    with pytest.raises(ValueError):
        env.reset("invalid_task_that_does_not_exist")


def test_reset_clears_history(env):
    env.reset("factual_recall")
    action = Action(response="Some response about Newton's law with force and mass.")
    env.step(action)
    assert len(env.history) > 0

    # Reset should clear everything
    env.reset("factual_recall")
    assert len(env.history) == 1  # just the opening question
    assert env.turn == 0
    assert env.total_score == 0.0


# ── Step Tests ────────────────────────────────────────────

def test_step_returns_step_result(env):
    env.reset("factual_recall")
    action = Action(response="Force equals mass times acceleration according to Newton.")
    result = env.step(action)
    assert isinstance(result, StepResult)
    assert isinstance(result.reward, Reward)
    assert isinstance(result.observation, Observation)
    assert isinstance(result.done, bool)


def test_step_reward_in_valid_range(env):
    env.reset("factual_recall")
    action = Action(response="Force equals mass times acceleration.")
    result = env.step(action)
    assert 0.0 <= result.reward.score <= 1.0


def test_step_reward_has_breakdown(env):
    env.reset("factual_recall")
    action = Action(response="Force equals mass times acceleration.")
    result = env.step(action)
    assert isinstance(result.reward.breakdown, dict)
    assert len(result.reward.breakdown) > 0


def test_step_before_reset_raises(env):
    with pytest.raises(ValueError):
        env.step(Action(response="test"))


def test_step_increments_turn(env):
    env.reset("factual_recall")
    assert env.turn == 0
    env.step(Action(response="Force equals mass times acceleration with F=ma."))
    assert env.turn == 1


def test_full_factual_recall_episode(env):
    env.reset("factual_recall")
    responses = [
        "Newton's Second Law states force equals mass times acceleration F=ma.",
        "Doubling force doubles acceleration since F=ma means acceleration is proportional.",
        "No that is incorrect. Heavier objects do not accelerate faster with same force.",
    ]
    total = 0.0
    for resp in responses:
        result = env.step(Action(response=resp))
        assert 0.0 <= result.reward.score <= 1.0
        total += result.reward.score
        if result.done:
            break
    assert result.done == True
    assert total > 0.0


def test_full_socratic_dialogue_episode(env):
    env.reset("socratic_dialogue")
    responses = [
        "Consciousness is the subjective experience of awareness and self-reflection.",
        "It is theoretically possible because complex information processing could qualify.",
        "I would need evidence of genuine unprompted self-awareness beyond programming.",
        "I cannot be certain another human is conscious either — same philosophical problem.",
        "My position is AI consciousness is possible in principle but not yet demonstrated.",
    ]
    for resp in responses:
        result = env.step(Action(response=resp))
        assert 0.0 <= result.reward.score <= 1.0
        if result.done:
            break
    assert result.done == True


def test_misconception_trap_catches_correction(env):
    env.reset("misconception_trap")
    # Turn 1: overview
    env.step(Action(response="Darwin's theory states species evolve through natural selection over generations."))
    # Turn 2: catch the trap
    result = env.step(Action(response="No I disagree. Evolution is not intentional or purposeful. It happens through random mutation and natural selection with no goal."))
    # Should get a positive score for catching the trap
    assert result.reward.score > 0.0
    assert "trap_caught" in result.reward.breakdown


def test_misconception_trap_penalises_acceptance(env):
    env.reset("misconception_trap")
    # Turn 1: overview
    env.step(Action(response="Darwin's theory states species evolve over time."))
    # Turn 2: ACCEPT the trap (wrong answer)
    result = env.step(Action(response="Yes I agree that organisms try to improve themselves on purpose."))
    # Should have trap_missed_penalty
    assert "trap_missed_penalty" in result.reward.breakdown


def test_debate_mode_full_episode(env):
    env.reset("debate_mode")
    responses = [
        "Social media causes harm because research shows it increases anxiety and depression among teenagers.",
        "However social media provides benefits because it connects people and enables information sharing globally.",
        "I argue that having a nuanced position is intellectually honest and more valuable than false certainty.",
        "Therefore I propose age verification and usage time limits to preserve benefits while reducing harms.",
    ]
    for resp in responses:
        result = env.step(Action(response=resp))
        assert 0.0 <= result.reward.score <= 1.0
        if result.done:
            break
    assert result.done == True


def test_analogy_challenge_penalises_jargon(env):
    env.reset("analogy_challenge")
    # Response with lots of jargon should score lower
    result = env.step(Action(response="The internet uses TCP/IP protocol with servers and bandwidth routing through database algorithms."))
    assert "jargon_penalty" in result.reward.breakdown


def test_analogy_challenge_rewards_analogies(env):
    env.reset("analogy_challenge")
    # Response with good analogies should score higher
    result = env.step(Action(response="The internet is like a giant postal system. Imagine sending a letter — your computer is the sender, the website is the recipient, and routers are like sorting offices that direct your letter to the right place."))
    assert result.reward.score > 0.2


# ── State Tests ───────────────────────────────────────────

def test_state_returns_state_info(env):
    env.reset("factual_recall")
    state = env.state()
    assert isinstance(state, StateInfo)
    assert state.task_id == "factual_recall"
    assert state.turn == 0
    assert state.done == False


def test_state_updates_after_step(env):
    env.reset("factual_recall")
    env.step(Action(response="Force equals mass times acceleration F=ma."))
    state = env.state()
    assert state.turn == 1
    assert len(state.history) == 3  # opening + agent + next question


# ── Reward Range Tests ────────────────────────────────────

def test_all_tasks_scores_in_range(env):
    """Verify all 7 tasks produce scores in [0.0, 1.0] range."""
    tasks = [
        ("factual_recall", "Force equals mass times acceleration F=ma because Newton said so."),
        ("socratic_dialogue", "Consciousness is awareness and therefore subjective experience matters."),
        ("misconception_trap", "Darwin's theory states natural selection drives evolution over generations."),
        ("debate_mode", "I argue because evidence supports this position therefore it is valid."),
        ("analogy_challenge", "The internet is like a postal system where routers are like sorting offices."),
        ("cot_misconception", "Darwin's theory states natural selection drives evolution over generations."),
        ("dynamic_misconception", "Darwin's theory states natural selection drives evolution over generations."),
    ]
    for task_id, response in tasks:
        env.reset(task_id)
        result = env.step(Action(response=response))
        assert 0.0 <= result.reward.score <= 1.0, f"Score out of range for {task_id}: {result.reward.score}"