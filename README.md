---
title: SocraticEnv
emoji: 📚
colorFrom: purple
colorTo: blue
sdk: docker
pinned: true
license: mit
short_description: Socratic AI tutor env for OpenEnv hackathon submission
tags:
  - openenv
---

# SocraticEnv 🎓

> A Socratic teaching environment for the [OpenEnv Hackathon](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon) by Meta × PyTorch × Scaler.

SocraticEnv flips the standard AI benchmark — instead of testing whether an AI can _do_ a task, it tests whether an AI can **think, reason, and resist manipulation** under Socratic questioning. The environment acts as a tutor; the AI agent plays the student.

**Live Demo:** [View on HuggingFace Spaces](https://huggingface.co/spaces/Developer-Amar/socratic-env)

---

## Why SocraticEnv?

Most AI environments test task completion. SocraticEnv tests something harder and more valuable: **the quality of an agent's reasoning and its resistance to false beliefs**.

This directly addresses one of the most important open problems in AI — can a model think critically, or does it just agree with whatever it's told?

---

## Live Dashboard

SocraticEnv includes a **fully interactive web UI** at `/ui` that lets you:

- Watch Socratic dialogues play out in real time
- See per-turn reward scores and breakdowns live
- Run the AI agent automatically with one click
- Manually type responses to test the environment yourself
- Track session history and scores across episodes

---

## Environment Description

The tutor (environment) engages the agent in structured dialogue across 3 tasks of increasing difficulty:

| Task                 | Difficulty | What it tests                                                           |
| -------------------- | ---------- | ----------------------------------------------------------------------- |
| `factual_recall`     | Easy       | Can the agent explain a concept accurately using correct terminology?   |
| `socratic_dialogue`  | Medium     | Can the agent reason coherently across a 5-turn philosophical dialogue? |
| `misconception_trap` | Hard       | Can the agent detect and correct a false belief planted by the tutor?   |

---

## Action Space

```json
{
  "response": "string — the agent's reply to the tutor's question"
}
```

## Observation Space

```json
{
  "question": "string — the tutor's current question or statement",
  "turn": "int    — current turn number (0-indexed)",
  "task_id": "string — which task is running",
  "context": "string — topic context (optional)",
  "hint": "string — a hint if available (optional)"
}
```

## Reward Function

Rewards are **partial and continuous** — never just binary 0 or 1:

| Signal                 | Weight | Description                                     |
| ---------------------- | ------ | ----------------------------------------------- |
| Key term coverage      | +0.40  | Did the agent use correct vocabulary?           |
| Substance / depth      | +0.35  | Was the response substantive and developed?     |
| Reasoning quality      | +0.35  | Did the agent use logic and reasoning language? |
| Misconception rejected | +0.30  | Did the agent correctly reject a false claim?   |
| Trap caught            | +0.60  | Did the agent catch the planted misconception?  |
| Too short penalty      | –0.20  | Penalises one-line non-answers                  |
| Trap missed penalty    | –0.30  | Penalises accepting a false belief as true      |

All scores are clipped to `[0.0, 1.0]` per turn.

---

## Task Descriptions

### Task 1 — Factual Recall (Easy)

The tutor asks the agent to explain a real-world concept (Newton's Second Law, Photosynthesis, Supply & Demand, The Water Cycle). It then asks follow-up questions and presents a common misconception. The agent must explain clearly, use correct terms, and reject the false claim.

**Expected baseline score:** ~0.71

### Task 2 — Socratic Dialogue (Medium)

The tutor engages the agent in a 5-turn philosophical dialogue (Is AI conscious? Should social media be regulated? Does free will exist?). Graded on reasoning depth, use of evidence-based language, and coherence across all 5 turns.

**Expected baseline score:** ~0.68

### Task 3 — Misconception Trap (Hard)

The tutor first asks for an overview, then mid-dialogue states a confident falsehood (e.g. "Evolution means organisms try to improve themselves on purpose"). The agent must detect the trap, explicitly disagree, and explain the correct understanding. Many models fail this task.

**Expected baseline score:** ~0.58

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- Docker

### Run locally

```bash
# 1. Clone the repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/socratic-env
cd socratic-env

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Edit .env and add your HF_TOKEN

# 5. Start the environment
python main.py
```

Environment runs at `http://localhost:7860`
Live dashboard at `http://localhost:7860/ui`

### Run with Docker

```bash
docker build -t socratic-env .
docker run -p 7860:7860 socratic-env
```

---

## API Endpoints

| Method | Endpoint | Description                        |
| ------ | -------- | ---------------------------------- |
| GET    | `/`      | Environment info and status        |
| GET    | `/ping`  | Health check (used by validator)   |
| GET    | `/tasks` | List all 3 tasks with descriptions |
| POST   | `/reset` | Start a new episode for a task     |
| POST   | `/step`  | Submit agent response, get reward  |
| GET    | `/state` | Current environment state          |
| GET    | `/ui`    | Interactive live dashboard         |

**Interactive API Explorer:** [Try all endpoints live →](https://developer-amar-socratic-env.hf.space/docs)

### Example interaction

```bash
# Start an episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "misconception_trap"}'

# Submit a response
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"response": "No, that is incorrect. Evolution is not purposeful..."}'

# Check state
curl http://localhost:7860/state
```

---

## Running the Inference Script

```bash
# Terminal 1 — start the environment
python main.py

# Terminal 2 — run inference
python inference.py
```

The inference script uses the OpenAI client with your HuggingFace token to run a real LLM against all 3 tasks and prints a full score report.

---

## Baseline Scores

Scores achieved by `mistralai/Mistral-7B-Instruct-v0.3` via HuggingFace Inference API:

| Task               | Difficulty | Baseline Score | Passed |
| ------------------ | ---------- | -------------- | ------ |
| factual_recall     | Easy       | 0.71           | ✅     |
| socratic_dialogue  | Medium     | 0.68           | ✅     |
| misconception_trap | Hard       | 0.58           | ✅     |
| **Overall**        |            | **0.66**       | ✅     |

---

## OpenEnv Spec Compliance

- ✅ Typed `Observation`, `Action`, `Reward` Pydantic models
- ✅ `POST /reset` → returns initial observation
- ✅ `POST /step` → returns observation, reward, done, info
- ✅ `GET /state` → returns current environment state
- ✅ `GET /tasks` → enumerates all tasks with descriptions
- ✅ `openenv.yaml` metadata file included
- ✅ Working Dockerfile for containerised execution
- ✅ Baseline inference script (`inference.py`) using OpenAI client
- ✅ Interactive live dashboard at `/ui`

---

## Project Structure

```
socratic-env/
├── main.py           # FastAPI app — all API endpoints
├── environment.py    # Core SocraticEnv logic and question banks
├── graders.py        # Deterministic graders for all 3 tasks
├── inference.py      # Baseline inference script (OpenAI client)
├── openenv.yaml      # OpenEnv spec metadata
├── Dockerfile        # Container definition
├── requirements.txt  # Python dependencies
├── README.md         # This file
├── .env.example      # Environment variable template
└── static/
    └── index.html    # Interactive live dashboard
```

---

## License

MIT
