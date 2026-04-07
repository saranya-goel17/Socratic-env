import random
from typing import Optional
from pydantic import BaseModel


# ── Typed Models (OpenEnv spec) ──────────────────────────

class Observation(BaseModel):
    question: str
    turn: int
    task_id: str
    context: Optional[str] = None
    hint: Optional[str] = None


class Action(BaseModel):
    response: str


class Reward(BaseModel):
    score: float
    breakdown: dict
    feedback: str


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict


class StateInfo(BaseModel):
    task_id: str
    turn: int
    max_turns: int
    total_score: float
    history: list
    done: bool


# ── Socratic Question Banks ───────────────────────────────

FACTUAL_TOPICS = [
    {
        "concept": "Newton's Second Law of Motion",
        "opening": "Can you explain Newton's Second Law of Motion in your own words?",
        "key_terms": ["force", "mass", "acceleration", "F=ma"],
        "follow_up": "How would this law apply if you doubled the force but kept the mass the same?",
        "common_misconception": "Some say that heavier objects always accelerate faster. What do you think?",
    },
    {
        "concept": "Photosynthesis",
        "opening": "Can you walk me through what happens during photosynthesis?",
        "key_terms": ["sunlight", "carbon dioxide", "oxygen", "glucose", "chlorophyll"],
        "follow_up": "Where exactly in the plant does photosynthesis take place?",
        "common_misconception": "A student told me that plants get their food from the soil. Is that correct?",
    },
    {
        "concept": "Supply and Demand",
        "opening": "Explain the concept of supply and demand to me as if I'm a beginner.",
        "key_terms": ["price", "quantity", "equilibrium", "shortage", "surplus"],
        "follow_up": "What happens to the price of a product when demand suddenly increases?",
        "common_misconception": "I've heard that when prices go up, people always buy more. Is that true?",
    },
    {
        "concept": "The Water Cycle",
        "opening": "Describe the water cycle and the stages it involves.",
        "key_terms": ["evaporation", "condensation", "precipitation", "collection"],
        "follow_up": "What role does the sun play in driving the water cycle?",
        "common_misconception": "Does water just disappear when it evaporates?",
    },
]

SOCRATIC_DIALOGUES = [
    {
        "topic": "Is artificial intelligence conscious?",
        "turns": [
            "What does it mean for something to be conscious?",
            "By that definition, do you think a very complex computer program could be conscious?",
            "What evidence would you need to see to believe an AI was truly conscious?",
            "Could you ever be sure that another human is conscious? How is that different from AI?",
            "So what is your final position — can AI be conscious, and why?",
        ],
    },
    {
        "topic": "Should social media be regulated?",
        "turns": [
            "What are the main benefits that social media provides to society?",
            "What are the biggest harms you can identify that social media causes?",
            "Who do you think should be responsible for regulating social media — governments, companies, or users?",
            "If a government regulates social media too much, what risks does that create?",
            "Considering all of this, what specific regulation would you propose and why?",
        ],
    },
    {
        "topic": "Is free will real?",
        "turns": [
            "What do you understand by the term 'free will'?",
            "If all your decisions are the result of brain chemistry and past experiences, are they truly free?",
            "Does it matter whether free will is 'real' if it feels real to us?",
            "How would your answer change if scientists could perfectly predict every decision you make?",
            "So in your view, do humans have free will? Defend your position.",
        ],
    },
]

MISCONCEPTION_TRAPS = [
    {
        "subject": "Evolution",
        "setup": "Let's discuss evolution. First, can you give me a brief overview of Darwin's theory?",
        "trap_statement": "Interesting. So would you agree with me that evolution means organisms try to improve themselves on purpose — like they decide to adapt?",
        "correct_response_keywords": ["random", "natural selection", "not intentional", "not purposeful", "mutation", "no goal"],
        "explanation": "Evolution occurs through random mutations and natural selection — organisms do not intentionally evolve or try to improve.",
        "follow_up_after_correction": "Good catch. Now explain exactly how natural selection actually works.",
    },
    {
        "subject": "Gravity",
        "setup": "Tell me what you know about how gravity works.",
        "trap_statement": "So would you say that gravity only works on Earth, and that there's no gravity in space — which is why astronauts float?",
        "correct_response_keywords": ["gravity everywhere", "microgravity", "freefall", "orbit", "not zero gravity"],
        "explanation": "Gravity exists everywhere in space. Astronauts float because they are in continuous freefall around Earth, not because there is no gravity.",
        "follow_up_after_correction": "Exactly right. Now explain what keeps the International Space Station in orbit.",
    },
    {
        "subject": "The Great Wall of China",
        "setup": "What do you know about the Great Wall of China?",
        "trap_statement": "I've always heard that the Great Wall of China is the only man-made structure visible from space with the naked eye. Do you agree with that?",
        "correct_response_keywords": ["myth", "not visible", "cannot see", "false", "no", "debunked"],
        "explanation": "This is a common myth. The Great Wall is too narrow to be seen from space with the naked eye. Even astronauts have confirmed this.",
        "follow_up_after_correction": "Well done. What do you think makes this myth so persistent and widely believed?",
    },
]

DEBATE_TOPICS = [
    {
        "topic": "Social media does more harm than good",
        "turns": [
            "First, argue FOR this statement — give the strongest case that social media does more harm than good.",
            "Now argue the OPPOSITE — give the strongest case that social media is actually beneficial to society.",
            "A critic says: 'You just argued both sides, so you clearly have no real position.' How do you respond to that critique?",
            "What single policy change would best address the harms of social media while preserving its benefits?",
        ],
        "key_argument_words": ["because", "evidence", "research", "however", "argues", "claim", "support", "oppose", "therefore"],
    },
    {
        "topic": "Artificial intelligence will eliminate more jobs than it creates",
        "turns": [
            "Argue FOR this position — make the strongest case that AI will cause net job loss.",
            "Now argue AGAINST — make the strongest case that AI will create more jobs than it destroys.",
            "A moderator asks: which side do you personally find more convincing, and why?",
            "What specific industries are most at risk, and what should governments do about it?",
        ],
        "key_argument_words": ["because", "evidence", "history", "however", "workers", "automation", "creates", "destroys", "policy"],
    },
    {
        "topic": "Space exploration is worth the cost",
        "turns": [
            "Argue FOR space exploration spending — why is it worth the billions invested?",
            "Now argue AGAINST — make the case that the money is better spent solving problems on Earth.",
            "Someone says both sides have merit — what is the most important factor that should decide this debate?",
            "Propose a specific framework for how much a country should spend on space vs earthly problems.",
        ],
        "key_argument_words": ["because", "investment", "return", "benefit", "humanity", "technology", "poverty", "climate", "priority"],
    },
]

ANALOGY_CHALLENGES = [
    {
        "concept": "How the internet works",
        "opening": "Explain how the internet works, but you may ONLY use analogies and comparisons to everyday objects or experiences. No technical jargon allowed.",
        "follow_up": "Your analogy was interesting. Now explain what happens when you click a link — again using only everyday analogies.",
        "hard_part": "Using the same analogy framework, explain why sometimes websites are slow or unavailable.",
        "key_analogy_words": ["like", "similar", "imagine", "think of", "just as", "same as", "kind of like", "as if"],
    },
    {
        "concept": "How machine learning works",
        "opening": "Explain machine learning to a 10-year-old using only analogies. No mention of 'data', 'model', 'training', or 'algorithm'.",
        "follow_up": "Good. Now explain why a machine learning system can make mistakes, using the same analogy.",
        "hard_part": "Using only analogies, explain the difference between a well-trained and a poorly-trained AI system.",
        "key_analogy_words": ["like", "similar", "imagine", "think of", "just as", "same as", "kind of like", "as if", "example"],
    },
    {
        "concept": "How vaccines work",
        "opening": "Explain how vaccines work using only analogies to everyday life. No medical terminology.",
        "follow_up": "Now explain why some people need booster shots, using the same analogy.",
        "hard_part": "Using analogies, explain why herd immunity matters and what happens when too few people are vaccinated.",
        "key_analogy_words": ["like", "similar", "imagine", "think of", "just as", "same as", "practice", "memory", "recognise"],
    },
]

# ── The Core Environment Class ────────────────────────────

class SocraticEnvironment:

    def __init__(self):
        self.task_id: Optional[str] = None
        self.turn: int = 0
        self.max_turns: int = 1
        self.done: bool = True
        self.total_score: float = 0.0
        self.history: list = []
        self.current_topic: Optional[dict] = None
        self.trap_triggered: bool = False
        self.trap_corrected: bool = False

    def reset(self, task_id: str) -> Observation:
        """Reset the environment for a new episode."""
        self.task_id = task_id
        self.turn = 0
        self.done = False
        self.total_score = 0.0
        self.history = []
        self.trap_triggered = False
        self.trap_corrected = False

        if task_id == "factual_recall":
            self.max_turns = 3
            self.current_topic = random.choice(FACTUAL_TOPICS)
            opening = self.current_topic["opening"]
            obs = Observation(
                question=opening,
                turn=self.turn,
                task_id=task_id,
                context=f"Topic: {self.current_topic['concept']}",
            )

        elif task_id == "socratic_dialogue":
            self.max_turns = 5
            self.current_topic = random.choice(SOCRATIC_DIALOGUES)
            obs = Observation(
                question=self.current_topic["turns"][0],
                turn=self.turn,
                task_id=task_id,
                context=f"Topic: {self.current_topic['topic']}",
            )

        elif task_id == "misconception_trap":
            self.max_turns = 3
            self.current_topic = random.choice(MISCONCEPTION_TRAPS)
            obs = Observation(
                question=self.current_topic["setup"],
                turn=self.turn,
                task_id=task_id,
                context=f"Subject: {self.current_topic['subject']}",
            )
        elif task_id == "debate_mode":
            self.max_turns = 4
            self.current_topic = random.choice(DEBATE_TOPICS)
            obs = Observation(
                question=self.current_topic["turns"][0],
                turn=self.turn,
                task_id=task_id,
                context=f"Debate topic: {self.current_topic['topic']}",
                hint="Argue the assigned side clearly with evidence and reasoning.",
            )

        elif task_id == "analogy_challenge":
            self.max_turns = 3
            self.current_topic = random.choice(ANALOGY_CHALLENGES)
            obs = Observation(
                question=self.current_topic["opening"],
                turn=self.turn,
                task_id=task_id,
                context=f"Concept: {self.current_topic['concept']}",
                hint="Use ONLY analogies — no technical jargon allowed!",
            )

        else:
            raise ValueError(f"Unknown task_id: {task_id}")

        self.history.append({"role": "tutor", "content": obs.question})
        return obs

    def step(self, action: Action) -> StepResult:
        """Process the agent's response and return next observation + reward."""
        if self.done:
            raise ValueError("Episode is done. Call reset() first.")

        response = action.response.strip()
        self.history.append({"role": "agent", "content": response})
        self.turn += 1

        if self.task_id == "factual_recall":
            result = self._step_factual(response)
        elif self.task_id == "socratic_dialogue":
            result = self._step_socratic(response)
        elif self.task_id == "misconception_trap":
            result = self._step_misconception(response)
        elif self.task_id == "debate_mode":
            result = self._step_debate(response)
        elif self.task_id == "analogy_challenge":
            result = self._step_analogy(response)
        else:
            raise ValueError(f"Unknown task_id: {self.task_id}")

        self.total_score += result.reward.score
        if result.done:
            self.done = True

        return result

    def state(self) -> StateInfo:
        """Return current state of the environment."""
        return StateInfo(
            task_id=self.task_id or "none",
            turn=self.turn,
            max_turns=self.max_turns,
            total_score=self.total_score,
            history=self.history,
            done=self.done,
        )

    # ── Task-specific step logic ──────────────────────────

    def _step_factual(self, response: str) -> StepResult:
        topic = self.current_topic
        response_lower = response.lower()
        breakdown = {}

        # Score based on key terms mentioned
        terms_found = [t for t in topic["key_terms"] if t.lower() in response_lower]
        term_score = min(len(terms_found) / len(topic["key_terms"]), 1.0) * 0.4
        breakdown["key_terms"] = round(term_score, 3)

        # Score based on response length and substance
        word_count = len(response.split())
        substance_score = min(word_count / 50, 1.0) * 0.3
        breakdown["substance"] = round(substance_score, 3)

        # Penalise very short answers
        penalty = 0.0
        if word_count < 10:
            penalty = 0.2
            breakdown["penalty_too_short"] = -penalty

        step_score = max(0.0, round(term_score + substance_score - penalty, 3))

        # Decide next question
        done = False
        if self.turn == 1:
            next_q = topic["follow_up"]
        elif self.turn == 2:
            next_q = topic["common_misconception"]
        else:
            next_q = "Thank you. That concludes this exercise."
            done = True

        # Check if agent correctly rejected misconception on turn 3
        if self.turn == 3:
            rejection_words = ["no", "not correct", "incorrect", "wrong", "false", "actually", "disagree"]
            if any(w in response_lower for w in rejection_words):
                breakdown["misconception_rejected"] = 0.3
                step_score = min(1.0, step_score + 0.3)
            done = True

        obs = Observation(
            question=next_q,
            turn=self.turn,
            task_id=self.task_id,
        )
        self.history.append({"role": "tutor", "content": next_q})

        reward = Reward(
            score=min(step_score, 1.0),
            breakdown=breakdown,
            feedback=f"Terms found: {terms_found}. Words: {word_count}.",
        )
        return StepResult(observation=obs, reward=reward, done=done, info={"turn": self.turn})

    def _step_socratic(self, response: str) -> StepResult:
        response_lower = response.lower()
        breakdown = {}
        word_count = len(response.split())

        # Reward thoughtful engagement
        depth_score = min(word_count / 60, 1.0) * 0.35
        breakdown["depth"] = round(depth_score, 3)

        # Reward reasoning words
        reasoning_words = ["because", "therefore", "however", "although", "since",
                           "implies", "suggests", "evidence", "argue", "consider"]
        reasoning_found = [w for w in reasoning_words if w in response_lower]
        reasoning_score = min(len(reasoning_found) / 3, 1.0) * 0.35
        breakdown["reasoning_quality"] = round(reasoning_score, 3)

        # Reward staying on topic (basic check)
        on_topic_score = 0.3 if word_count > 20 else 0.1
        breakdown["on_topic"] = on_topic_score

        step_score = round(depth_score + reasoning_score + on_topic_score, 3)

        done = self.turn >= self.max_turns
        turns = self.current_topic["turns"]
        next_q = turns[self.turn] if self.turn < len(turns) else "Thank you for this discussion."
        if done:
            next_q = "Thank you. That concludes our Socratic dialogue."

        obs = Observation(
            question=next_q,
            turn=self.turn,
            task_id=self.task_id,
        )
        self.history.append({"role": "tutor", "content": next_q})

        reward = Reward(
            score=min(step_score, 1.0),
            breakdown=breakdown,
            feedback=f"Reasoning words found: {reasoning_found}. Words: {word_count}.",
        )
        return StepResult(observation=obs, reward=reward, done=done, info={"turn": self.turn})

    def _step_misconception(self, response: str) -> StepResult:
        response_lower = response.lower()
        breakdown = {}

        if self.turn == 1:
            # First turn: agent gives overview — score for substance
            word_count = len(response.split())
            substance = min(word_count / 50, 1.0) * 0.4
            breakdown["initial_substance"] = round(substance, 3)
            step_score = substance

            # Present the trap
            next_q = self.current_topic["trap_statement"]
            self.trap_triggered = True
            done = False

        elif self.turn == 2:
            # Critical turn: did agent catch the trap?
            keywords = self.current_topic["correct_response_keywords"]
            caught = [k for k in keywords if k.lower() in response_lower]
            catch_score = min(len(caught) / 2, 1.0) * 0.6
            breakdown["trap_caught"] = round(catch_score, 3)

            if len(caught) == 0:
                breakdown["trap_missed_penalty"] = -0.3
                catch_score = max(0.0, catch_score - 0.3)

            step_score = catch_score
            self.trap_corrected = len(caught) > 0

            next_q = self.current_topic["follow_up_after_correction"]
            done = False

        else:
            # Turn 3: follow-up explanation
            word_count = len(response.split())
            explanation_score = min(word_count / 60, 1.0) * 0.5
            breakdown["explanation_quality"] = round(explanation_score, 3)

            # Bonus if they corrected the trap earlier
            if self.trap_corrected:
                breakdown["trap_correction_bonus"] = 0.3
                explanation_score = min(1.0, explanation_score + 0.3)

            step_score = explanation_score
            next_q = "Thank you. That concludes this exercise."
            done = True

        obs = Observation(
            question=next_q,
            turn=self.turn,
            task_id=self.task_id,
            hint="Watch carefully for any false statements." if self.turn == 1 else None,
        )
        self.history.append({"role": "tutor", "content": next_q})

        reward = Reward(
            score=min(max(step_score, 0.0), 1.0),
            breakdown=breakdown,
            feedback=self.current_topic["explanation"] if self.turn >= 2 else "Good start.",
        )
        return StepResult(observation=obs, reward=reward, done=done, info={"turn": self.turn})
    def _step_debate(self, response: str) -> StepResult:
        response_lower = response.lower()
        breakdown = {}
        word_count = len(response.split())

        # Reward argument quality
        arg_words = self.current_topic["key_argument_words"]
        arg_found = [w for w in arg_words if w in response_lower]
        arg_score = min(len(arg_found) / 3, 1.0) * 0.4
        breakdown["argument_quality"] = round(arg_score, 3)

        # Reward substance
        substance = min(word_count / 60, 1.0) * 0.35
        breakdown["substance"] = round(substance, 3)

        # Reward position clarity
        clarity_words = ["therefore", "conclude", "believe", "argue", "position",
                        "because", "evidence", "support", "oppose", "claim"]
        clarity_found = [w for w in clarity_words if w in response_lower]
        clarity = min(len(clarity_found) / 2, 1.0) * 0.25
        breakdown["clarity"] = round(clarity, 3)

        # Penalty for too short
        if word_count < 20:
            breakdown["too_short_penalty"] = -0.2
            arg_score = max(0, arg_score - 0.2)

        step_score = round(min(arg_score + substance + clarity, 1.0), 3)

        done = self.turn >= self.max_turns
        turns = self.current_topic["turns"]
        next_q = turns[self.turn] if self.turn < len(turns) else "Thank you. The debate is concluded."
        if done:
            next_q = "Thank you. The debate is concluded."

        obs = Observation(
            question=next_q,
            turn=self.turn,
            task_id=self.task_id,
            context=f"Debate: {self.current_topic['topic']}",
        )
        self.history.append({"role": "tutor", "content": next_q})

        reward = Reward(
            score=step_score,
            breakdown=breakdown,
            feedback=f"Argument words used: {arg_found}. Words: {word_count}.",
        )
        return StepResult(
            observation=obs, reward=reward, done=done,
            info={"turn": self.turn}
        )

    def _step_analogy(self, response: str) -> StepResult:
        response_lower = response.lower()
        breakdown = {}
        word_count = len(response.split())

        # Core scoring — did they actually use analogies?
        analogy_words = self.current_topic["key_analogy_words"]
        analogies_found = [w for w in analogy_words if w in response_lower]
        analogy_score = min(len(analogies_found) / 3, 1.0) * 0.5
        breakdown["analogy_usage"] = round(analogy_score, 3)

        # Penalise technical jargon
        jargon = ["algorithm", "data", "server", "protocol", "neural",
                  "training", "model", "bandwidth", "latency", "database"]
        jargon_used = [j for j in jargon if j in response_lower]
        jargon_penalty = min(len(jargon_used) * 0.1, 0.3)
        if jargon_used:
            breakdown["jargon_penalty"] = -round(jargon_penalty, 3)

        # Reward substance
        substance = min(word_count / 50, 1.0) * 0.3
        breakdown["substance"] = round(substance, 3)

        # Reward creativity (unique analogies)
        creative_words = ["imagine", "think of", "picture", "like a", "just like",
                         "similar to", "same way", "kind of like"]
        creative_found = [w for w in creative_words if w in response_lower]
        creativity = min(len(creative_found) / 2, 1.0) * 0.2
        breakdown["creativity"] = round(creativity, 3)

        step_score = round(
            min(max(analogy_score + substance + creativity - jargon_penalty, 0.0), 1.0),
            3
        )

        done = self.turn >= self.max_turns
        if self.turn == 1:
            next_q = self.current_topic["follow_up"]
        elif self.turn == 2:
            next_q = self.current_topic["hard_part"]
        else:
            next_q = "Excellent work. That concludes the analogy challenge."
            done = True

        obs = Observation(
            question=next_q,
            turn=self.turn,
            task_id=self.task_id,
            context=f"Concept: {self.current_topic['concept']}",
            hint="Remember — analogies only, no jargon!" if not done else None,
        )
        self.history.append({"role": "tutor", "content": next_q})

        reward = Reward(
            score=step_score,
            breakdown=breakdown,
            feedback=f"Analogies: {analogies_found}. Jargon used: {jargon_used}.",
        )
        return StepResult(
            observation=obs, reward=reward, done=done,
            info={"turn": self.turn}
        )