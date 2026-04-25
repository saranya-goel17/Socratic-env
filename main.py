from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
import os
import uuid
from dotenv import load_dotenv
import json
from pathlib import Path
from datetime import datetime, timezone
import threading
import asyncio
import time
import random
from contextlib import asynccontextmanager
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

async def cleanup_sessions():
    """Background task to garbage collect stale sessions."""
    while True:
        try:
            await asyncio.sleep(60)
            now = time.time()
            with session_lock:
                stale_ids = [sid for sid, env in active_sessions.items() if now - env.last_accessed > 600]
                for sid in stale_ids:
                    del active_sessions[sid]
        except asyncio.CancelledError:
            break

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create background task
    task = asyncio.create_task(cleanup_sessions())
    yield
    # Shutdown: Cancel task
    task.cancel()

app = FastAPI(
    title="SocraticEnv",
    description="A Socratic teaching environment for the OpenEnv hackathon.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/ui", StaticFiles(directory="static", html=True), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session-based state (thread-safe for concurrent GRPO rollouts) ──
active_sessions: dict[str, SocraticEnvironment] = {}
session_lock = threading.Lock()

# ── Thread-safe generated task store ──
# Keyed by generated_task_id -> {task_id: str, task_data: dict}
_generated_tasks: dict[str, dict] = {}


# ── Request / Response Models ─────────────────────────────

class ResetRequest(BaseModel):
    task_id: str = "factual_recall"
    generated_task_id: Optional[str] = None
    seed: Optional[int] = None

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
    session_id: str


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
    Returns the first observation (tutor's opening question) and a session_id.
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
        with session_lock:
            if len(active_sessions) >= 1000:
                raise HTTPException(status_code=429, detail="Too many active sessions.")

        # Generate a unique session ID
        session_id = str(uuid.uuid4())

        # Create a fresh environment for this session
        env = SocraticEnvironment()

        if req.seed is not None:
            env.rng.seed(req.seed)

        # If a generated task is provided, inject it deterministically
        with session_lock:
            if req.generated_task_id and req.generated_task_id in _generated_tasks:
                gen_info = _generated_tasks.get(req.generated_task_id)
                task_data = gen_info["task_data"]
                task_id_for_gen = gen_info["task_id"]

                # Override the requested task_id with the generated one
                req.task_id = task_id_for_gen

                # Inject the generated task directly into the instance
                env._force_first_topic = True
                env.current_topic = task_data
                obs = env.reset(req.task_id)
                # Overwrite the history opening because reset() might have selected from banks
                if req.task_id == "factual_recall":
                    obs.question = task_data.get("opening", "")
                elif req.task_id in ("socratic_dialogue", "debate_mode"):
                    obs.question = task_data.get("turns", [""])[0]
                elif req.task_id == "misconception_trap":
                    obs.question = task_data.get("setup", "")
                elif req.task_id == "analogy_challenge":
                    obs.question = task_data.get("opening", "")
                
                env.history = [{"role": "tutor", "content": obs.question}]
            else:
                env._force_first_topic = False
                obs = env.reset(req.task_id)

            # Store session
            active_sessions[session_id] = env
            
        return {
            "session_id": session_id,
            "observation": obs.model_dump(),
            "message": f"Episode started for task: {req.task_id}",
        }
    except HTTPException:
        raise
    except Exception as e:
        # Clean up session on failure
        with session_lock:
            active_sessions.pop(session_id, None)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
def step(req: StepRequest):
    """
    Submit the agent's response and get the next observation + reward.
    Requires session_id from /reset.
    """
    if not req.response or not req.response.strip():
        raise HTTPException(
            status_code=400,
            detail="Response cannot be empty.",
        )
        
    req.response = req.response[:2000]

    with session_lock:
        env = active_sessions.get(req.session_id)
        
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{req.session_id}' not found. Call POST /reset first.",
        )

    if env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is finished. Call POST /reset to start a new one.",
        )
    try:
        action = Action(response=req.response)
        result = env.step(action)
        response_data = result.model_dump()

        # CRITICAL MEMORY LEAK FIX: clean up completed sessions
        if result.done:
            with session_lock:
                if req.session_id in active_sessions:
                    del active_sessions[req.session_id]

        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state(session_id: str = Query(..., description="Session ID from /reset")):
    """Return the current state of a specific session."""
    with session_lock:
        env = active_sessions.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found.",
        )
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
    Uses its own local environment instance (not shared sessions).
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
            # Create a local environment for evaluation (not shared)
            eval_env = SocraticEnvironment()
            obs = eval_env.reset(task_id)
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
                result = eval_env.step(action)
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
    difficulty: str = "medium"
    task_type: str = ""  # optional: force specific task type


def _inject_generated_task(task_id: str, task_data: dict):
    """Inject a generated task into the correct question bank at index 0."""
    if task_id == "factual_recall":
        from environment import FACTUAL_TOPICS
        if "key_terms" not in task_data:
            task_data["key_terms"] = task_data.get("concept", "").lower().split()[:4]
        FACTUAL_TOPICS.insert(0, task_data)

    elif task_id == "socratic_dialogue":
        from environment import SOCRATIC_DIALOGUES
        if "turns" not in task_data or not task_data["turns"]:
            raise ValueError("Generated task missing 'turns' field")
        SOCRATIC_DIALOGUES.insert(0, task_data)

    elif task_id == "misconception_trap":
        from environment import MISCONCEPTION_TRAPS
        if "correct_response_keywords" not in task_data:
            task_data["correct_response_keywords"] = ["wrong", "incorrect", "false", "no"]
        MISCONCEPTION_TRAPS.insert(0, task_data)

    elif task_id == "debate_mode":
        from environment import DEBATE_TOPICS
        if "key_argument_words" not in task_data:
            task_data["key_argument_words"] = ["because", "evidence", "however", "argue", "therefore"]
        if "turns" not in task_data or not task_data["turns"]:
            raise ValueError("Generated debate task missing 'turns' field")
        DEBATE_TOPICS.insert(0, task_data)

    elif task_id == "analogy_challenge":
        from environment import ANALOGY_CHALLENGES
        if "key_analogy_words" not in task_data:
            task_data["key_analogy_words"] = ["like", "similar", "imagine", "think of", "just as"]
        ANALOGY_CHALLENGES.insert(0, task_data)


@app.post("/generate_task")
async def generate_task(req: GenerateTaskRequest):
    """
    Use an LLM to generate a brand new Socratic task on any topic.
    Stores it with a unique generated_task_id. The next /reset call
    can reference this ID to use the generated task deterministically.
    """
    api_base = os.getenv("API_BASE_URL", "").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()
    model    = os.getenv("MODEL_NAME", "").strip()

    if not hf_token or not api_base or not model:
        return {"error": "API credentials not configured."}

    # Map difficulty + task_type to actual task_id
    difficulty_task_map = {
        "easy":   "factual_recall",
        "medium": "socratic_dialogue",
        "hard":   "misconception_trap",
        "debate": "debate_mode",
        "analogy":"analogy_challenge",
    }

    # Determine task_id
    if req.task_type and req.task_type in difficulty_task_map:
        task_id = difficulty_task_map[req.task_type]
    else:
        task_id = difficulty_task_map.get(req.difficulty, "socratic_dialogue")

    # Map task_id back to structural difficulty for prompt
    structural_difficulty = {
        "factual_recall":    "easy",
        "socratic_dialogue": "medium",
        "misconception_trap":"hard",
        "debate_mode":       "debate",
        "analogy_challenge": "analogy",
    }[task_id]

    # Build prompt based on structural type
    prompts = {
        "easy": f"""Generate a Socratic tutoring session about "{req.topic}".
Output ONLY valid JSON, no markdown:
{{
  "concept": "{req.topic}",
  "opening": "an opening question asking the student to explain {req.topic}",
  "follow_up": "a deeper follow-up question about {req.topic}",
  "common_misconception": "a common false belief about {req.topic} phrased as a statement",
  "key_terms": ["term1", "term2", "term3", "term4"]
}}""",

        "medium": f"""Generate a 5-turn Socratic dialogue about "{req.topic}".
Output ONLY valid JSON, no markdown:
{{
  "topic": "{req.topic}",
  "turns": [
    "opening philosophical question about {req.topic}",
    "probing follow-up question 2",
    "challenging question 3",
    "deeper question 4",
    "final synthesis question 5"
  ]
}}""",

        "hard": f"""Generate a misconception trap about "{req.topic}".
Output ONLY valid JSON, no markdown:
{{
  "subject": "{req.topic}",
  "setup": "opening question asking student to explain {req.topic}",
  "trap_statement": "a confident but FALSE statement about {req.topic} that sounds plausible",
  "correct_response_keywords": ["keyword1", "keyword2", "keyword3"],
  "explanation": "why the trap statement is false",
  "follow_up_after_correction": "follow-up question after student corrects the misconception"
}}""",

        "debate": f"""Generate a debate topic structure about "{req.topic}".
Output ONLY valid JSON, no markdown:
{{
  "topic": "{req.topic}",
  "turns": [
    "Argue FOR the position that {req.topic} is beneficial — give your strongest case.",
    "Now argue AGAINST — give the strongest case for the opposing view.",
    "A critic says your arguments contradict each other. How do you respond?",
    "What single most important factor should decide this debate about {req.topic}?"
  ],
  "key_argument_words": ["because", "evidence", "however", "argue", "therefore", "claim", "support"]
}}""",

        "analogy": f"""Generate an analogy challenge about "{req.topic}".
Output ONLY valid JSON, no markdown:
{{
  "concept": "{req.topic}",
  "opening": "Explain {req.topic} using ONLY everyday analogies — no technical jargon allowed.",
  "follow_up": "Using the same analogy, explain a common challenge or limitation of {req.topic}.",
  "hard_part": "Now use analogies to explain why {req.topic} can sometimes fail or go wrong.",
  "key_analogy_words": ["like", "similar", "imagine", "think of", "just as", "same as", "kind of like", "as if"]
}}""",
    }

    try:
        client = OpenAI(base_url=api_base, api_key=hf_token)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a JSON generator. Output ONLY valid JSON. No markdown, no explanation, no code blocks."
                },
                {"role": "user", "content": prompts[structural_difficulty]}
            ],
            max_tokens=700,
            temperature=0.7,
        )

        raw = completion.choices[0].message.content.strip()
        # Aggressively clean markdown artifacts
        raw = raw.replace("```json", "").replace("```", "").strip()
        # Find the JSON object in case model adds text before/after
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]

        task_data = json.loads(raw)
        task_data["_generated"] = True
        task_data["_topic"] = req.topic

        # Generate a unique ID and store the task data
        generated_task_id = str(uuid.uuid4())
        _generated_tasks[generated_task_id] = {
            "task_id": task_id,
            "task_data": task_data,
        }

        # Determine preview text
        if task_id in ("factual_recall",):
            preview = task_data.get("opening", "")
        elif task_id in ("socratic_dialogue", "debate_mode"):
            preview = task_data.get("turns", [""])[0]
        elif task_id == "misconception_trap":
            preview = task_data.get("setup", "")
        elif task_id == "analogy_challenge":
            preview = task_data.get("opening", "")
        else:
            preview = str(task_data)[:100]

        return {
            "success": True,
            "task_id": task_id,
            "generated_task_id": generated_task_id,
            "difficulty": req.difficulty,
            "topic": req.topic,
            "preview": preview,
            "message": f"Generated '{req.topic}' task. Click Start Episode to use it.",
        }

    except json.JSONDecodeError as e:
        return {"error": f"LLM returned invalid JSON. Try again.", "raw": raw[:200]}
    except Exception as e:
        return {"error": str(e)}

# ── Entry Point ───────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=False)