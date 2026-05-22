"""
ReActAgent — Reasoning + Acting loop for agentic RAG.

What is ReAct?
  ReAct (Reason + Act) is a pattern where the LLM alternates between:
    - Thought: reasoning about what to do next
    - Action: calling a tool to get information
    - Observation: the tool's result
  This loop continues until the LLM decides it has enough information
  to give a final Answer.

Why ReAct for RAG?
  A simple RAG pipeline always does the same thing: search → generate.
  ReAct lets the agent decide dynamically:
    - Is this a knowledge question? → search the KB
    - Is this a stats question? → query the database
    - Is the KB empty? → scrape a page first, then search
    - Can I answer from what I already know? → answer directly

Loop structure:
  Thought: I need to find information about X
  Action: search_knowledge_base("X")
  Observation: [retrieved chunks]
  Thought: I have enough context to answer
  Answer: [final response]

  Max iterations = MAX_STEPS to prevent infinite loops.

Usage:
    agent = ReActAgent()
    result = agent.run("What is the Raft consensus algorithm?")
    print(result.answer)
    for step in result.steps:
        print(step)
"""

import re
import ollama
from dataclasses import dataclass, field

from agent.tools import AgentTools, ToolResult
from store.vector_store import VectorStore
from store.sql_store import SQLStore
from embeddings.embedder import Embedder

LLM_MODEL = "llama3.1:8b"

# Maximum number of Thought → Action → Observation loops before forcing an answer
MAX_STEPS = 5

# System prompt defining the ReAct loop format for the LLM
REACT_SYSTEM_PROMPT = """You are a helpful research assistant with access to a knowledge base of scraped articles.

You answer questions using a Thought → Action → Observation loop.

Available actions:
- search_knowledge_base(query) — search for relevant content in the knowledge base
- query_database(question) — get statistics about stored articles
- scrape_and_ingest(url) — scrape a new URL and add it to the knowledge base
- get_kb_stats() — see what's in the knowledge base
- answer(response) — give your final answer (end the loop)

Rules:
- Always start with a Thought explaining what you plan to do
- Call exactly one action per step
- Use search_knowledge_base for conceptual questions
- Use query_database for counting or filtering questions
- Use answer() when you have enough information to respond
- If the knowledge base has no relevant content, say so honestly

Format each step EXACTLY like this:
Thought: <your reasoning>
Action: <action_name>(<argument>)

When ready to answer:
Thought: I have enough information to answer.
Action: answer(<your complete response>)"""


@dataclass
class AgentStep:
    """A single step in the ReAct loop."""
    thought: str
    action: str       # e.g. "search_knowledge_base"
    action_input: str # e.g. "What is CAP theorem?"
    observation: str  # tool result

    def __str__(self) -> str:
        return (
            f"Thought:     {self.thought}\n"
            f"Action:      {self.action}({self.action_input!r})\n"
            f"Observation: {self.observation[:200]}{'...' if len(self.observation) > 200 else ''}"
        )


@dataclass
class AgentResult:
    """Full output of a ReAct agent run."""
    question: str
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    stopped_early: bool = False  # True if MAX_STEPS was hit

    def __str__(self) -> str:
        step_strs = "\n\n".join(f"Step {i+1}:\n{s}" for i, s in enumerate(self.steps))
        note = " (stopped at max steps)" if self.stopped_early else ""
        return (
            f"Q: {self.question}\n\n"
            f"--- Reasoning trace ---\n{step_strs}\n\n"
            f"--- Answer{note} ---\n{self.answer}"
        )


class ReActAgent:
    """
    ReAct agent that uses tools to answer questions about the knowledge base.

    Args:
        model:       Ollama model for reasoning (default: llama3.1:8b)
        tools:       AgentTools instance (created fresh if not provided)
        max_steps:   Maximum reasoning steps before forcing an answer

    Usage:
        agent = ReActAgent()
        result = agent.run("What is the Raft consensus algorithm?")
        print(result.answer)
        for step in result.steps:
            print(step)
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        tools: AgentTools | None = None,
        max_steps: int = MAX_STEPS,
    ):
        self.model = model
        self.tools = tools or AgentTools()
        self.max_steps = max_steps

    def run(self, question: str) -> AgentResult:
        """
        Run the ReAct loop to answer the question.

        Alternates between LLM reasoning and tool calls until the LLM
        calls answer() or we hit max_steps.

        Args:
            question: The user's question

        Returns:
            AgentResult with final answer and full reasoning trace
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        # Conversation history — grows as the loop progresses
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ]

        steps = []

        for step_num in range(self.max_steps):
            # Ask the LLM for the next Thought + Action
            response = self._call_llm(messages)

            # Parse the LLM's response into thought + action
            thought, action, action_input = self._parse_response(response)

            # If the LLM chose to answer, we're done
            if action == "answer":
                steps_so_far = steps  # capture before returning
                return AgentResult(
                    question=question,
                    answer=action_input,
                    steps=steps,
                )

            # Execute the tool
            observation = self._execute_tool(action, action_input)

            # Record this step
            step = AgentStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            )
            steps.append(step)

            # Add the step to the conversation so the LLM has context
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}\n\nContinue."
            })

        # Hit max steps — force a final answer
        forced_answer = self._force_answer(question, steps, messages)
        return AgentResult(
            question=question,
            answer=forced_answer,
            steps=steps,
            stopped_early=True,
        )

    def _call_llm(self, messages: list[dict]) -> str:
        """Call Ollama and return the response text."""
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={"temperature": 0.1},
        )
        return response.message.content.strip()

    def _parse_response(self, response: str) -> tuple[str, str, str]:
        """
        Parse the LLM's Thought + Action output.

        Expected format:
          Thought: <text>
          Action: <tool_name>(<argument>)

        Returns (thought, action_name, action_input).
        Falls back to "answer" if parsing fails.
        """
        thought = ""
        action = "answer"
        action_input = response  # fallback: treat full response as answer

        # Extract Thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Extract Action: tool_name(argument)
        action_match = re.search(r"Action:\s*(\w+)\((.+?)\)\s*$", response, re.DOTALL | re.MULTILINE)
        if action_match:
            action = action_match.group(1).strip()
            action_input = action_match.group(2).strip().strip('"').strip("'")

        return thought, action, action_input

    def _execute_tool(self, action: str, action_input: str) -> str:
        """
        Route an action name to the corresponding AgentTools method.

        Returns the tool's output string, or an error message if the
        action is unknown or the tool fails.
        """
        tool_map = {
            "search_knowledge_base": lambda: self.tools.search_knowledge_base(action_input),
            "query_database":        lambda: self.tools.query_database(action_input),
            "scrape_and_ingest":     lambda: self.tools.scrape_and_ingest(action_input),
            "get_kb_stats":          lambda: self.tools.get_kb_stats(),
        }

        if action not in tool_map:
            return f"Unknown tool: {action!r}. Available: {list(tool_map.keys())}"

        result: ToolResult = tool_map[action]()
        return result.output if result.success else f"Tool error: {result.error}"

    def _force_answer(self, question: str, steps: list[AgentStep], messages: list[dict]) -> str:
        """
        Ask the LLM for a final answer using whatever context was gathered.

        Called when MAX_STEPS is reached without the LLM calling answer().
        """
        context = "\n\n".join(s.observation for s in steps if s.observation)
        prompt = (
            f"Based on everything gathered so far, answer the original question.\n\n"
            f"Question: {question}\n\n"
            f"Context gathered:\n{context or 'None'}\n\n"
            f"Answer:"
        )
        messages.append({"role": "user", "content": prompt})
        return self._call_llm(messages)
