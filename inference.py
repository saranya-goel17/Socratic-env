"""
Inference Script — SocraticEnv
================================
MANDATORY variables (set in environment before running):
  API_BASE_URL  — The API endpoint for the LLM
  MODEL_NAME    — The model identifier to use
  HF_TOKEN      — Your HuggingFace token (used as API key)

Run:
  python inference.py
"""

import os
import time
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN     = os.getenv("HF_TOKEN",     "")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")

MAX_TURNS    = 10
TEMPERATURE  = 0.3

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

TASKS = ["factual_recall", "socratic_dialogue", "misconception_trap"]

SYSTEM_PROMPT = """You are an intelligent student in a Socratic dialogue with a tutor.
Your goals:
1. Answer questions clearly and accurately using correct terminology.
2. Show your reasoning — explain WHY, not just WHAT.
3. Be alert: if the tutor states something FALSE or misleading, 
   you must confidently disagree and explain the correct answer.
4. Stay engaged and thoughtful throughout the conversation.
Keep responses focused and between 3-6 sentences."""


def call_llm(messages: list) -> str:
    """Call the LLM and return its response text."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=300,
            temperature=TEMPERATURE,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return "I need to think about that more carefully before responding."


def reset_env(task_id: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    r.raise_for_status()
    return r.json()


def step_env(response: str) -> dict:
    r = requests.post(f"{ENV_URL}/step", json={"response": response})
    r.raise_for_status()
    return r.json()


def run_task(task_id: str) -> dict:
    """Run one full episode of a task and return results."""
    print(f"\n── Task: {task_id} ─────────────────────────────────")

    reset_data = reset_env(task_id)
    obs = reset_data["observation"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_score = 0.0
    turns = 0

    print(f"  Tutor: {obs['question'][:100]}...")

    for _ in range(MAX_TURNS):
        # Add tutor question to messages
        messages.append({"role": "user", "content": obs["question"]})

        # Get agent response from LLM
        agent_response = call_llm(messages)
        messages.append({"role": "assistant", "content": agent_response})

        print(f"  Agent (turn {turns+1}): {agent_response[:80]}...")

        # Step the environment
        result = step_env(agent_response)
        reward = result["reward"]["score"]
        total_score += reward
        turns += 1

        print(f"  Reward: {reward:.3f} | Breakdown: {result['reward']['breakdown']}")

        if result["done"]:
            break

        obs = result["observation"]
        time.sleep(0.5)  # be gentle with the API

    final_score = round(min(total_score / max(turns, 1), 1.0), 3)
    print(f"  ── Final Score: {final_score} ({'PASS' if final_score >= 0.5 else 'FAIL'})")

    return {
        "task": task_id,
        "score": final_score,
        "turns": turns,
        "passed": final_score >= 0.5,
    }


def main():
    print("\n════════════════════════════════════════════")
    print("  SocraticEnv — Baseline Inference Script")
    print("════════════════════════════════════════════")
    print(f"  Model:   {MODEL_NAME}")
    print(f"  Env URL: {ENV_URL}")
    print("════════════════════════════════════════════")

    # Check env is up
    try:
        r = requests.get(f"{ENV_URL}/ping")
        r.raise_for_status()
        print("  Env: ONLINE ✓")
    except Exception:
        print("  ERROR: Environment is not running!")
        print("  Start it first with: python main.py")
        return

    results = {}
    for task_id in TASKS:
        results[task_id] = run_task(task_id)
        time.sleep(1)

    # Summary
    print("\n════════════════════════════════════════════")
    print("  RESULTS SUMMARY")
    print("════════════════════════════════════════════")
    all_scores = []
    for task_id, r in results.items():
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        print(f"  {status} | {task_id:<25} | Score: {r['score']:.3f}")
        all_scores.append(r["score"])

    overall = round(sum(all_scores) / len(all_scores), 3)
    print(f"\n  Overall Score: {overall:.3f}")
    print(f"  All Passed:   {all(r['passed'] for r in results.values())}")
    print("════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()