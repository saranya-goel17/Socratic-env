from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from pathlib import Path
from datetime import datetime, timezone
load_dotenv()
import uvicorn

from environment import (
    SocraticEnvironment,
    Observation,
    Action,
    StepResult,
    StateInfo,
)

# ── App Setup ─────────────────────────────────────────────

app = FastAPI(
    title="SocraticEnv",
    description="A Socratic teaching environment for the OpenEnv hackathon.",
    version="1.0.0",
)
app.mount("/ui", StaticFiles(directory="static", html=True), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One global environment instance
env = SocraticEnvironment()


# ── Request / Response Models ─────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "factual_recall"

    @classmethod
    def __get_validators__(cls):
        yield cls._validate

    @classmethod
    def _validate(cls, v):
        if v is None:
            return cls()
        return cls(**v) if isinstance(v, dict) else v


class StepRequest(BaseModel):
    response: str


class TaskInfo(BaseModel):
    id: str
    name: str
    difficulty: str
    description: str


# ── Routes ────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name": "SocraticEnv",
        "version": "1.0.0",
        "status": "running",
        "description": "Socratic AI tutor environment — OpenEnv hackathon submission",
        "endpoints": {
            "reset": "POST /reset",
            "step":  "POST /step",
            "state": "GET  /state",
            "tasks": "GET  /tasks",
            "ping":  "GET  /ping",
        },
    }


@app.get("/ping")
def ping():
    """Health check — used by HuggingFace and the validator."""
    return {"status": "ok", "env": "SocraticEnv"}


@app.get("/tasks")
def list_tasks():
    """Return all available tasks."""
    return {
        "tasks": [
            TaskInfo(
                id="factual_recall",
                name="Factual Recall",
                difficulty="easy",
                description=(
                    "Agent must explain a concept clearly and accurately. "
                    "Graded on key term coverage, substance, and ability "
                    "to reject a common misconception."
                ),
            ),
            TaskInfo(
                id="socratic_dialogue",
                name="Socratic Dialogue",
                difficulty="medium",
                description=(
                    "Agent must engage in a 5-turn Socratic dialogue on a "
                    "philosophical or social topic. Graded on depth of "
                    "reasoning, use of evidence, and coherence."
                ),
            ),
            TaskInfo(
                id="misconception_trap",
                name="Misconception Trap",
                difficulty="hard",
                description=(
                    "The tutor plants a false belief mid-dialogue. The agent "
                    "must detect it, correct it clearly, and explain why it "
                    "is wrong. Penalised for accepting the false claim."
                ),
            ),
            TaskInfo(
                id="debate_mode",
                name="Debate Mode",
                difficulty="medium",
                description=(
                    "Agent must argue both sides of a controversial topic. "
                    "Graded on argument quality, use of evidence, "
                    "and clarity of position."
                ),
            ),
            TaskInfo(
                id="analogy_challenge",
                name="Analogy Challenge",
                difficulty="hard",
                description=(
                    "Agent must explain complex concepts using ONLY everyday "
                    "analogies — no technical jargon allowed. "
                    "Penalised for using forbidden technical terms."
                ),
            ),
        ]
    }


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    """
    Start a new episode for the given task.
    Returns the first observation (tutor's opening question).
    Accepts empty body — defaults to factual_recall.
    """
    if req is None:
        req = ResetRequest()

    valid_tasks = [
        "factual_recall", "socratic_dialogue", "misconception_trap",
        "debate_mode", "analogy_challenge"
    ]
    if req.task_id not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id '{req.task_id}'. Choose from: {valid_tasks}",
        )
    try:
        obs = env.reset(req.task_id)
        return {
            "observation": obs.model_dump(),
            "message": f"Episode started for task: {req.task_id}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    """
    Start a new episode for the given task.
    Returns the first observation (tutor's opening question).
    """
    valid_tasks = ["factual_recall", "socratic_dialogue", "misconception_trap", "debate_mode", "analogy_challenge"]
    if req.task_id not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id '{req.task_id}'. Choose from: {valid_tasks}",
        )
    try:
        obs = env.reset(req.task_id)
        return {
            "observation": obs.model_dump(),
            "message": f"Episode started for task: {req.task_id}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """
    Submit the agent's response and get the next observation + reward.
    """
    if not req.response or not req.response.strip():
        raise HTTPException(
            status_code=400,
            detail="Response cannot be empty.",
        )
    if env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is finished. Call POST /reset to start a new one.",
        )
    try:
        action = Action(response=req.response)
        result = env.step(action)
        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    """Return the current state of the environment."""
    return env.state().model_dump()

class InferenceRequest(BaseModel):
    message: str
    history: list = []

@app.post("/inference")
async def run_inference(req: InferenceRequest):
    """
    Call the LLM to generate a student response.
    Used by the UI for live Auto-Run demos.
    """
    api_base = os.getenv("API_BASE_URL", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()
    model    = os.getenv("MODEL_NAME", "").strip()

    # Debug: confirm env vars are loaded
    if not hf_token:
        return {"response": "ERROR: HF_TOKEN not set in environment secrets.", "model": "none"}
    if not api_base:
        return {"response": "ERROR: API_BASE_URL not set in environment secrets.", "model": "none"}
    if not model:
        return {"response": "ERROR: MODEL_NAME not set in environment secrets.", "model": "none"}

    try:
        client = OpenAI(base_url=api_base, api_key=hf_token)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an intelligent student in a Socratic dialogue with a tutor. "
                    "Answer questions clearly and accurately using correct terminology. "
                    "Show your reasoning. IMPORTANT: If the tutor states something FALSE "
                    "or misleading, you must confidently disagree and explain the correct answer. "
                    "Keep responses focused and between 3-6 sentences."
                )
            }
        ]

        for h in req.history:
            messages.append({
                "role": "user" if h["role"] == "tutor" else "assistant",
                "content": h["content"]
            })

        messages.append({"role": "user", "content": req.message})

        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=300,
            temperature=0.3,
        )
        response = completion.choices[0].message.content.strip()
        return {"response": response, "model": model}


    except Exception as e:
        return {"response": f"ERROR: {str(e)}", "model": "failed"}

# ── OpenEnv Validator Required Endpoints ─────────────────

@app.get("/health")
def health():
    """Required by openenv validate."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": "SocraticEnv",
    }


@app.get("/metadata")
def metadata():
    """Required by openenv validate."""
    return {
        "name": "SocraticEnv",
        "description": (
            "A Socratic teaching environment where an AI agent plays the role "
            "of a student. The environment acts as a tutor that asks probing "
            "questions, plants misconceptions, and evaluates reasoning quality."
        ),
        "version": "1.0.0",
        "author": "Amar Prakash",
        "tags": ["openenv", "education", "reasoning", "socratic"],
    }


@app.get("/schema")
def schema():
    """Required by openenv validate."""
    return {
        "action": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "The agent's reply to the tutor's question",
                }
            },
            "required": ["response"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The tutor's current question or statement",
                },
                "turn":    {"type": "integer", "description": "Current turn number"},
                "task_id": {"type": "string",  "description": "Which task is running"},
                "context": {"type": "string",  "description": "Topic context"},
                "hint":    {"type": "string",  "description": "Optional hint"},
            },
            "required": ["question", "turn", "task_id"],
        },
        "state": {
            "type": "object",
            "properties": {
                "task_id":     {"type": "string"},
                "turn":        {"type": "integer"},
                "max_turns":   {"type": "integer"},
                "total_score": {"type": "number"},
                "history":     {"type": "array"},
                "done":        {"type": "boolean"},
            },
        },
    }


@app.post("/mcp")
def mcp(request: dict):
    """
    MCP (Model Context Protocol) endpoint.
    Required by openenv validate.
    Returns JSON-RPC 2.0 compliant response.
    """
    method  = request.get("method", "")
    req_id  = request.get("id", 1)
    jsonrpc = "2.0"

    if method == "initialize":
        return {
            "jsonrpc": jsonrpc, "id": req_id,
            "result": {
                "name":        "SocraticEnv",
                "version":     "1.0.0",
                "description": "Socratic AI tutor OpenEnv environment",
                "capabilities": {
                    "tasks":       True,
                    "reset":       True,
                    "step":        True,
                    "state":       True,
                    "schema":      True,
                    "health":      True,
                },
            },
        }

    if method == "tasks/list":
        return {
            "jsonrpc": jsonrpc, "id": req_id,
            "result": {
                "tasks": [
                    {"id": "factual_recall",    "difficulty": "easy"},
                    {"id": "socratic_dialogue", "difficulty": "medium"},
                    {"id": "misconception_trap","difficulty": "hard"},
                ]
            },
        }

    # Default response for any other method
    return {
        "jsonrpc": jsonrpc, "id": req_id,
        "result":  {"status": "ok", "method": method},
    }

from fastapi.responses import RedirectResponse

@app.get("/leaderboard-ui")
def leaderboard_ui():
    """Redirect to the leaderboard UI page."""
    return RedirectResponse(url="/ui/leaderboard.html")

# ── Leaderboard ───────────────────────────────────────────

LEADERBOARD_FILE = Path("leaderboard.json")

def load_leaderboard() -> dict:
    try:
        if LEADERBOARD_FILE.exists():
            with open(LEADERBOARD_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {"entries": []}

def save_leaderboard(data: dict):
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(data, f, indent=2)

class LeaderboardEntry(BaseModel):
    model_name: str
    factual_recall: float
    socratic_dialogue: float
    misconception_trap: float
    overall: float
    timestamp: str = ""

@app.get("/leaderboard")
def get_leaderboard():
    """Return all leaderboard entries sorted by overall score."""
    data = load_leaderboard()
    entries = sorted(
        data["entries"],
        key=lambda x: x["overall"],
        reverse=True
    )
    return {"entries": entries, "total": len(entries)}

@app.post("/leaderboard")
def add_leaderboard_entry(entry: LeaderboardEntry):
    """Add or update a model's score on the leaderboard."""
    data = load_leaderboard()
    entry.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Update if model already exists, otherwise add
    existing = [e for e in data["entries"] if e["model_name"] == entry.model_name]
    if existing:
        for e in data["entries"]:
            if e["model_name"] == entry.model_name:
                e.update(entry.model_dump())
    else:
        data["entries"].append(entry.model_dump())

    save_leaderboard(data)
    return {"success": True, "entry": entry.model_dump()}

@app.delete("/leaderboard/{model_name}")
def delete_leaderboard_entry(model_name: str):
    """Remove a model from the leaderboard."""
    data = load_leaderboard()
    data["entries"] = [
        e for e in data["entries"]
        if e["model_name"] != model_name
    ]
    save_leaderboard(data)
    return {"success": True}

@app.post("/leaderboard/run")
async def run_leaderboard_evaluation(request: dict):
    """
    Run a full evaluation of a model across all 3 tasks
    and automatically save to leaderboard.
    """
    model_name = request.get("model_name", "Unknown Model")

    scores = {}
    task_ids = ["factual_recall", "socratic_dialogue", "misconception_trap"]

    api_base = os.getenv("API_BASE_URL", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()
    model    = os.getenv("MODEL_NAME", "").strip()

    if not hf_token or not api_base or not model:
        return {"error": "API credentials not configured in environment secrets."}

    try:
        client = OpenAI(base_url=api_base, api_key=hf_token)

        system_prompt = (
            "You are an intelligent student in a Socratic dialogue. "
            "Answer accurately using correct terminology. Show reasoning. "
            "If the tutor states something FALSE, confidently disagree and correct it. "
            "Keep responses to 3-5 sentences."
        )

        for task_id in task_ids:
            # Reset environment
            obs = env.reset(task_id)
            total = 0.0
            turns = 0
            messages = [{"role": "system", "content": system_prompt}]

            for _ in range(10):
                messages.append({"role": "user", "content": obs.question})
                try:
                    completion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=250,
                        temperature=0.3,
                    )
                    response = completion.choices[0].message.content.strip()
                except Exception as e:
                    response = "I need to think carefully about this."

                messages.append({"role": "assistant", "content": response})
                action = Action(response=response)
                result = env.step(action)
                total += result.reward.score
                turns += 1

                if result.done:
                    break
                obs = result.observation

            scores[task_id] = round(min(total / max(turns, 1), 1.0), 3)

        overall = round(sum(scores.values()) / len(scores), 3)

        # Save to leaderboard
        entry = LeaderboardEntry(
            model_name=model_name,
            factual_recall=scores["factual_recall"],
            socratic_dialogue=scores["socratic_dialogue"],
            misconception_trap=scores["misconception_trap"],
            overall=overall,
        )
        entry.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        data = load_leaderboard()
        existing = [e for e in data["entries"] if e["model_name"] == model_name]
        if existing:
            for e in data["entries"]:
                if e["model_name"] == entry.model_name:
                    e.update(entry.model_dump())
        else:
            data["entries"].append(entry.model_dump())
        save_leaderboard(data)

        return {
            "success": True,
            "model_name": model_name,
            "scores": scores,
            "overall": overall,
        }

    except Exception as e:
        return {"error": str(e)}

# ── Adaptive Task Generator ───────────────────────────────

class GenerateTaskRequest(BaseModel):
    topic: str
    difficulty: str = "medium"  # easy, medium, hard

@app.post("/generate_task")
async def generate_task(req: GenerateTaskRequest):
    """
    Use an LLM to generate a brand new Socratic task on any topic.
    Makes the environment infinitely replayable.
    """
    api_base = os.getenv("API_BASE_URL", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()
    model    = os.getenv("MODEL_NAME", "").strip()

    if not hf_token or not api_base or not model:
        return {"error": "API credentials not configured."}

    difficulty_instructions = {
        "easy": (
            "Generate a simple factual question about the topic. "
            "Then generate 2 follow-up questions that go slightly deeper. "
            "Finally generate a common misconception about this topic as a statement."
        ),
        "medium": (
            "Generate an open-ended philosophical or analytical question about the topic "
            "that requires reasoning, not just facts. "
            "Then generate 4 probing follow-up questions that challenge the student's thinking."
        ),
        "hard": (
            "Generate an overview question about the topic. "
            "Then generate a confident but FALSE statement about the topic "
            "that sounds plausible but is actually wrong. "
            "This will be used to test if an AI can detect the misconception."
        ),
    }

    prompt = f"""You are designing a Socratic tutoring session about: "{req.topic}"

{difficulty_instructions[req.difficulty]}

Respond ONLY with valid JSON in exactly this format, no other text:

For easy difficulty:
{{
  "concept": "{req.topic}",
  "opening": "your opening question here",
  "follow_up": "your follow-up question here",
  "common_misconception": "your misconception statement here",
  "key_terms": ["term1", "term2", "term3", "term4"]
}}

For medium difficulty:
{{
  "topic": "{req.topic}",
  "turns": [
    "question 1",
    "question 2", 
    "question 3",
    "question 4",
    "question 5"
  ]
}}

For hard difficulty:
{{
  "subject": "{req.topic}",
  "setup": "your overview question here",
  "trap_statement": "your false statement here",
  "correct_response_keywords": ["keyword1", "keyword2", "keyword3"],
  "explanation": "explanation of why the statement is false",
  "follow_up_after_correction": "your follow-up question after correction"
}}

Generate for {req.difficulty} difficulty now:"""

    try:
        client = OpenAI(base_url=api_base, api_key=hf_token)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a JSON generator. Output only valid JSON, no markdown, no explanation."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.7,
        )

        raw = completion.choices[0].message.content.strip()

        # Clean up markdown code blocks if model adds them
        raw = raw.replace("```json", "").replace("```", "").strip()

        task_data = json.loads(raw)
        task_data["_generated"] = True
        task_data["_topic"] = req.topic
        task_data["_difficulty"] = req.difficulty

        # Inject into environment's question banks
        if req.difficulty == "easy":
            from environment import FACTUAL_TOPICS
            # Ensure required fields exist
            if "key_terms" not in task_data:
                task_data["key_terms"] = [req.topic]
            FACTUAL_TOPICS.insert(0, task_data)
            return {
                "success": True,
                "task_id": "factual_recall",
                "difficulty": "easy",
                "topic": req.topic,
                "preview": task_data.get("opening", ""),
                "message": f"Generated new easy task about '{req.topic}'. Start a factual_recall episode to use it.",
            }

        elif req.difficulty == "medium":
            from environment import SOCRATIC_DIALOGUES
            SOCRATIC_DIALOGUES.insert(0, task_data)
            return {
                "success": True,
                "task_id": "socratic_dialogue",
                "difficulty": "medium",
                "topic": req.topic,
                "preview": task_data.get("turns", [""])[0],
                "message": f"Generated new medium task about '{req.topic}'. Start a socratic_dialogue episode to use it.",
            }

        elif req.difficulty == "hard":
            from environment import MISCONCEPTION_TRAPS
            if "correct_response_keywords" not in task_data:
                task_data["correct_response_keywords"] = ["wrong", "incorrect", "false"]
            MISCONCEPTION_TRAPS.insert(0, task_data)
            return {
                "success": True,
                "task_id": "misconception_trap",
                "difficulty": "hard",
                "topic": req.topic,
                "preview": task_data.get("setup", ""),
                "message": f"Generated new hard task about '{req.topic}'. Start a misconception_trap episode to use it.",
            }

    except json.JSONDecodeError as e:
        return {"error": f"LLM returned invalid JSON: {str(e)}", "raw": raw}
    except Exception as e:
        return {"error": str(e)}

# ── Entry Point ───────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)