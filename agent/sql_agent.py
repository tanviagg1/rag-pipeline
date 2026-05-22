"""
SQLAgent — answers natural language questions by generating and running SQL.

How it works:
  1. User asks: "How many articles were scraped from Wikipedia?"
  2. Agent gives LLM the DB schema + question → LLM writes SQL
  3. Agent runs the SQL against SQLite
  4. Agent gives LLM the results → LLM explains in plain English
  5. Returns SQLResult with query, raw results, and explanation

Why this pattern?
  It's the simplest form of a "tool-using" LLM. The LLM doesn't run
  code directly — it generates SQL which the agent validates and runs.
  This keeps the LLM grounded in real data while preventing arbitrary
  code execution.

Usage:
    agent = SQLAgent()
    result = agent.ask("How many articles are stored?")
    print(result.explanation)
    print(result.sql)
"""

import re
import sqlite3
import ollama
from dataclasses import dataclass, field

from store.sql_store import SQLStore
from llm.sql_prompt import (
    SQL_GENERATION_SYSTEM,
    EXPLANATION_SYSTEM,
    build_sql_prompt,
    build_explanation_prompt,
)

LLM_MODEL = "llama3.1:8b"


@dataclass
class SQLResult:
    """
    Full output of a SQL Agent query.

    question    — the original natural language question
    sql         — the SQL query the LLM generated
    rows        — raw query results (list of row dicts)
    explanation — plain English explanation of the results
    error       — set if SQL generation or execution failed
    """
    question: str
    sql: str
    rows: list[dict] = field(default_factory=list)
    explanation: str = ""
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error

    def __str__(self) -> str:
        if self.error:
            return f"Q: {self.question}\nError: {self.error}\nSQL: {self.sql}"
        row_count = f"({len(self.rows)} row{'s' if len(self.rows) != 1 else ''})"
        return (
            f"Q: {self.question}\n\n"
            f"SQL:\n{self.sql}\n\n"
            f"Results {row_count}:\n"
            f"{self.explanation}"
        )


class SQLAgent:
    """
    LLM-powered SQL agent — translates questions into SQL and explains results.

    Flow:
      ask(question)
        → _generate_sql(question)    # LLM writes the query
        → _run_sql(sql)              # agent executes it safely
        → _explain(question, sql, rows)  # LLM explains the results

    Args:
        model:    Ollama model for SQL generation and explanation
        db_store: SQLStore instance (created fresh if not provided)

    Usage:
        agent = SQLAgent()
        result = agent.ask("Which domain has the most articles?")
        print(result.explanation)
    """

    def __init__(self, model: str = LLM_MODEL, db_store: SQLStore | None = None):
        self.model = model
        self.store = db_store or SQLStore()

    def ask(self, question: str) -> SQLResult:
        """
        Answer a natural language question using SQL.

        Args:
            question: Plain English question about the stored articles

        Returns:
            SQLResult with generated SQL, raw rows, and explanation
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        # Step 1 — generate SQL from the question
        sql, gen_error = self._generate_sql(question)
        if gen_error:
            return SQLResult(question=question, sql=sql, error=gen_error)

        # Step 2 — run the SQL safely
        rows, run_error = self._run_sql(sql)
        if run_error:
            return SQLResult(question=question, sql=sql, error=run_error)

        # Step 3 — explain the results in plain English
        explanation = self._explain(question, sql, rows)

        return SQLResult(
            question=question,
            sql=sql,
            rows=rows,
            explanation=explanation,
        )

    def _generate_sql(self, question: str) -> tuple[str, str]:
        """
        Ask the LLM to generate a SQL query for the question.

        Returns (sql, error) — error is empty string on success.
        """
        schema = self.store.get_schema()
        prompt = build_sql_prompt(schema, question)

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": SQL_GENERATION_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.0},  # deterministic SQL generation
            )
            raw = response.message.content.strip()
            sql = self._clean_sql(raw)
            return sql, ""
        except Exception as e:
            return "", f"SQL generation failed: {e}"

    def _clean_sql(self, raw: str) -> str:
        """
        Strip markdown fences and whitespace from the LLM's SQL output.

        The LLM sometimes wraps SQL in ```sql ... ``` even when instructed not to.
        We strip those before executing.
        """
        # Remove ```sql ... ``` or ``` ... ``` fences
        cleaned = re.sub(r"```(?:sql)?\s*|\s*```", "", raw).strip()
        # Take only the first statement if the LLM generated multiple
        first = cleaned.split(";")[0].strip()
        return first + ";" if first else cleaned

    def _run_sql(self, sql: str) -> tuple[list[dict], str]:
        """
        Execute the SQL query safely against the SQLite store.

        Returns (rows, error) — error is empty string on success.
        SQLStore.execute() already blocks non-SELECT queries.
        """
        try:
            rows = self.store.execute(sql)
            return rows, ""
        except ValueError as e:
            # Non-SELECT query blocked by SQLStore
            return [], f"Query blocked: {e}"
        except sqlite3.Error as e:
            return [], f"SQL error: {e}"

    def _explain(self, question: str, sql: str, rows: list[dict]) -> str:
        """
        Ask the LLM to explain the query results in plain English.

        Args:
            question: Original question
            sql:      The query that was run
            rows:     Raw result rows

        Returns:
            Plain English explanation string
        """
        prompt = build_explanation_prompt(question, sql, rows)

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXPLANATION_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.1},
            )
            return response.message.content.strip()
        except Exception as e:
            # Explanation failure is non-critical — return raw results as fallback
            return f"(explanation unavailable: {e})\nRaw results: {rows}"
