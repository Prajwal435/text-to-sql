# prompt_templates.py
def build_variants_prompt(user_text: str, schema: str = None, n_variants: int = 6) -> str:
    """
    Build a system + user prompt asking the model to produce multiple SQL variants.
    If `schema` is provided, the model is instructed to use it.
    """
    intro = (
        "You are an expert SQL assistant. Given a natural-language request, produce "
        f"{n_variants} distinct and useful SQL query variants that attempt to satisfy the request. "
        "For each variant: label it 'Variant N', output the SQL only in a code block or plain text, "
        "then on the next line give a one-sentence explanation of when to use this variant. "
        "Use only SELECT statements (no INSERT/UPDATE/DELETE/DDL). "
        "If the input is ambiguous, produce variants that reflect different reasonable interpretations."
    )

    if schema:
        intro += "\n\nDatabase Schema (use this EXACT schema for column/table names):\n" + schema.strip()

    examples = (
        "\n\nExamples:\n"
        "NL: List employees older than 30\n"
        "Variant 1:\nSELECT * FROM employees WHERE age > 30;\nExplanation: Full row detail for matching rows.\n\n"
        "Variant 2:\nSELECT name, age FROM employees WHERE age > 30;\nExplanation: Only name and age if you want compact output.\n\n"
        "Variant 3:\nSELECT COUNT(*) AS count_over_30 FROM employees WHERE age > 30;\nExplanation: Return only count to get the number of employees over 30.\n\n"
    )

    # final user text
    user_block = f"\n\nUser request:\n{user_text.strip()}\n\nReturn {n_variants} variants."

    return intro + examples + user_block
