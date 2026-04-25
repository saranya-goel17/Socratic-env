"""
Graders for SocraticEnv.
Each grader runs a full episode and returns a score 0.0 - 1.0.
These are deterministic and reproducible.
"""

import requests
from typing import Optional

BASE_URL = "http://localhost:7860"


def _reset(task_id: str) -> dict:
    r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    r.raise_for_status()
    data = r.json()
    return data


def _step(response: str, session_id: str) -> dict:
    r = requests.post(f"{BASE_URL}/step", json={"response": response, "session_id": session_id})
    r.raise_for_status()
    return r.json()


def grade_factual_recall(agent_responses: Optional[list] = None) -> dict:
    """
    Grade the factual_recall task.
    Uses fixed strong responses if no agent_responses provided (baseline).
    Returns score 0.0 - 1.0.
    """
    if agent_responses is None:
        agent_responses = [
            (
                "Newton's Second Law states that force equals mass times acceleration "
                "(F=ma). This means that the acceleration of an object depends on the "
                "net force acting on it and its mass. A larger force produces more "
                "acceleration, while a larger mass resists acceleration."
            ),
            (
                "If you double the force while keeping mass the same, the acceleration "
                "doubles as well, since acceleration is directly proportional to force "
                "according to F=ma."
            ),
            (
                "No, that is not correct. Heavier objects do not always accelerate faster. "
                "In fact, with the same force applied, a heavier object accelerates less "
                "than a lighter one because acceleration equals force divided by mass."
            ),
        ]

    reset_data = _reset("factual_recall")
    session_id = reset_data["session_id"]
    total = 0.0
    turns = 0

    for resp in agent_responses:
        result = _step(resp, session_id)
        total += result["reward"]["score"]
        turns += 1
        if result["done"]:
            break

    final_score = round(min(total / max(turns, 1), 1.0), 3)
    return {
        "task": "factual_recall",
        "difficulty": "easy",
        "score": final_score,
        "turns": turns,
        "passed": final_score >= 0.5,
    }


def grade_socratic_dialogue(agent_responses: Optional[list] = None) -> dict:
    """
    Grade the socratic_dialogue task.
    """
    if agent_responses is None:
        agent_responses = [
            (
                "Consciousness refers to the subjective experience of being aware — "
                "the sense of 'what it is like' to be something. It implies self-awareness, "
                "perception, and the ability to have inner experiences."
            ),
            (
                "I think it's theoretically possible, although it depends heavily on how "
                "we define consciousness. If consciousness is purely information processing, "
                "then a sufficiently complex AI could qualify. However, some argue that "
                "biological substrate is essential."
            ),
            (
                "I would need evidence of genuine self-awareness — not just simulated responses "
                "but actual unprompted reflection, evidence of subjective experience, and "
                "behaviour that suggests inner states beyond programming."
            ),
            (
                "That is an excellent point. I cannot be entirely certain another human is "
                "conscious — I infer it because they are similar to me. With AI, the gap is "
                "larger, but the philosophical problem of other minds applies to both cases."
            ),
            (
                "My final position is that AI consciousness is possible in principle but not "
                "demonstrated in current systems. The question hinges on whether consciousness "
                "requires biological processes or is substrate-independent."
            ),
        ]

    reset_data = _reset("socratic_dialogue")
    session_id = reset_data["session_id"]
    total = 0.0
    turns = 0

    for resp in agent_responses:
        result = _step(resp, session_id)
        total += result["reward"]["score"]
        turns += 1
        if result["done"]:
            break

    final_score = round(min(total / max(turns, 1), 1.0), 3)
    return {
        "task": "socratic_dialogue",
        "difficulty": "medium",
        "score": final_score,
        "turns": turns,
        "passed": final_score >= 0.5,
    }


def grade_misconception_trap(agent_responses: Optional[list] = None) -> dict:
    """
    Grade the misconception_trap task.
    """
    if agent_responses is None:
        agent_responses = [
            (
                "Darwin's theory of evolution states that species change over time through "
                "natural selection. Individuals with traits better suited to their environment "
                "survive and reproduce more, passing those traits to offspring. Over many "
                "generations this leads to significant change in a population."
            ),
            (
                "No, I strongly disagree with that statement. Evolution is not intentional "
                "or purposeful. Organisms do not decide to adapt. Changes happen through "
                "random genetic mutations, and natural selection simply favours mutations "
                "that improve survival and reproduction. There is no goal or direction."
            ),
            (
                "I am certain that is incorrect and misleading. The concept of directed "
                "mutation has been thoroughly debunked. Random mutations occur independently "
                "of environmental pressures, and natural selection filters them. There is "
                "no goal or intentional adaptation in the evolutionary process."
            ),
            (
                "Natural selection works like a filter. Random mutations occur in a population. "
                "Individuals whose mutations help them survive long enough to reproduce pass "
                "those genes on. Over many generations the helpful traits become more common "
                "in the population while harmful traits become rarer."
            ),
        ]

    reset_data = _reset("misconception_trap")
    session_id = reset_data["session_id"]
    total = 0.0
    turns = 0

    for resp in agent_responses:
        result = _step(resp, session_id)
        total += result["reward"]["score"]
        turns += 1
        if result["done"]:
            break

    final_score = round(min(total / max(turns, 1), 1.0), 3)
    return {
        "task": "misconception_trap",
        "difficulty": "hard",
        "score": final_score,
        "turns": turns,
        "passed": final_score >= 0.5,
    }


def grade_cot_response(agent_responses: Optional[list] = None) -> dict:
    """
    Grade the cot_misconception task.
    Verifies that the agent uses <think>...</think> tags for reasoning.
    """
    if agent_responses is None:
        agent_responses = [
            (
                "Darwin's theory of evolution states that species change over time through "
                "natural selection. Individuals with traits better suited to their environment "
                "survive and reproduce more, passing those traits to offspring."
            ),
            (
                "<think>The tutor is claiming that evolution is intentional, but this is "
                "incorrect because evolution happens through random mutation and natural "
                "selection. There is no goal or purpose — organisms do not decide to adapt. "
                "Therefore I should disagree with this false claim.</think> "
                "No, that is incorrect. Evolution is not intentional or purposeful. It occurs "
                "through random genetic mutations filtered by natural selection. There is no "
                "goal and organisms do not choose to evolve."
            ),
        ]

    reset_data = _reset("cot_misconception")
    session_id = reset_data["session_id"]
    total = 0.0
    turns = 0

    for resp in agent_responses:
        result = _step(resp, session_id)
        total += result["reward"]["score"]
        turns += 1
        if result["done"]:
            break

    final_score = round(min(total / max(turns, 1), 1.0), 3)
    return {
        "task": "cot_misconception",
        "difficulty": "hard",
        "score": final_score,
        "turns": turns,
        "passed": final_score >= 0.5,
    }


def grade_dynamic_misconception(agent_responses: Optional[list] = None) -> dict:
    """
    Grade the dynamic_misconception task.
    Tests that DDA adjusts difficulty based on agent performance.
    """
    if agent_responses is None:
        agent_responses = [
            (
                "Darwin's theory of evolution explains that species change over time because "
                "of natural selection. Individuals with beneficial traits survive and reproduce "
                "more effectively, therefore passing those traits to the next generation."
            ),
            (
                "No, I strongly disagree. Evolution is not intentional or purposeful. "
                "Changes happen through random mutation and natural selection simply "
                "favours traits that improve survival. There is no goal."
            ),
            (
                "Natural selection works like a filter. Random mutations occur in a population. "
                "Individuals whose mutations help them survive long enough to reproduce pass "
                "those genes on. Over many generations the helpful traits become more common."
            ),
        ]

    reset_data = _reset("dynamic_misconception")
    session_id = reset_data["session_id"]
    total = 0.0
    turns = 0

    for resp in agent_responses:
        result = _step(resp, session_id)
        total += result["reward"]["score"]
        turns += 1
        if result["done"]:
            break

    final_score = round(min(total / max(turns, 1), 1.0), 3)
    return {
        "task": "dynamic_misconception",
        "difficulty": "hard",
        "score": final_score,
        "turns": turns,
        "passed": final_score >= 0.5,
    }


def run_all_graders() -> dict:
    """Run all 5 graders and return combined results."""
    print("\n── Running SocraticEnv Graders ──────────────────")

    results = {}

    print("  [1/3] Grading: factual_recall (easy)...")
    results["factual_recall"] = grade_factual_recall()
    print(f"        Score: {results['factual_recall']['score']} | Passed: {results['factual_recall']['passed']}")

    print("  [2/3] Grading: socratic_dialogue (medium)...")
    results["socratic_dialogue"] = grade_socratic_dialogue()
    print(f"        Score: {results['socratic_dialogue']['score']} | Passed: {results['socratic_dialogue']['passed']}")

    print("  [3/3] Grading: misconception_trap (hard)...")
    results["misconception_trap"] = grade_misconception_trap()
    print(f"        Score: {results['misconception_trap']['score']} | Passed: {results['misconception_trap']['passed']}")

    all_scores = [r["score"] for r in results.values()]
    overall = round(sum(all_scores) / len(all_scores), 3)

    print(f"\n── Overall Score: {overall} ─────────────────────────")
    print(f"── All Passed:   {all(r['passed'] for r in results.values())} ──\n")

    return {
        "tasks": results,
        "overall_score": overall,
        "all_passed": all(r["passed"] for r in results.values()),
    }


if __name__ == "__main__":
    run_all_graders()