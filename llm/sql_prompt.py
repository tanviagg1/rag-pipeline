"""
SQL prompt templates for the SQL Agent.

The LLM needs two things to generate correct SQL:
  1. The database schema (table names, column names, types)
  2. The user's question in plain English

We give it both in a structured system prompt and ask it to respond
with ONLY the SQL query — no explanation, no markdown fences.
The agent then runs that query and asks the LLM to explain the results.

Usage:
    prompt = build_sql_prompt(schema, "How many articles are from Wikipedia?")
    # → string ready to send to the LLM
"""

# System prompt for SQL generation.
# Instructs the LLM to produce clean, runnable SQL with no extras.
SQL_GENERATION_SYSTEM = """You are an expert SQL assistant. You write SQLite SELECT queries.

Rules:
- Respond with ONLY the SQL query — no explanation, no markdown, no code fences
- Use only SQLite-compatible syntax
- Only use columns that exist in the schema provided
- Always use LIMIT when the result could be large (default LIMIT 20)
- For text searches, use LIKE with % wildcards
- Never use INSERT, UPDATE, DELETE, DROP, or any write operations"""


# System prompt for result explanation.
# After running the query, we ask the LLM to explain results in plain English.
EXPLANATION_SYSTEM = """You are a helpful assistant that explains database query results in plain English.
Keep your explanation concise and factual. Focus on what the data shows."""


def build_sql_prompt(schema: str, question: str) -> str:
    """
    Build the user prompt for SQL generation.

    Combines the DB schema and the user's question into a single prompt
    that gives the LLM enough context to write a correct query.

    Args:
        schema:   Database schema string (from SQLStore.get_schema())
        question: User's natural language question

    Returns:
        Formatted prompt string ready to send to the LLM
    """
    return f"""Database schema:
{schema}

Question: {question}

Write a SQLite SELECT query to answer this question."""


def build_explanation_prompt(question: str, sql: str, results: list[dict]) -> str:
    """
    Build the prompt for result explanation.

    After running the SQL query, we pass the question + query + raw results
    to the LLM and ask it to explain what the data shows.

    Args:
        question: The original user question
        sql:      The SQL query that was run
        results:  Raw query results (list of row dicts)

    Returns:
        Formatted prompt string
    """
    # Format results as a readable table string
    if not results:
        result_str = "(no rows returned)"
    elif len(results) <= 10:
        # Show all rows for small result sets
        rows = [str(dict(r)) for r in results]
        result_str = "\n".join(rows)
    else:
        # Show first 10 rows + count for large result sets
        rows = [str(dict(r)) for r in results[:10]]
        result_str = "\n".join(rows) + f"\n... ({len(results)} total rows)"

    return f"""Question: {question}

SQL query used:
{sql}

Query results:
{result_str}

Explain these results in plain English."""
