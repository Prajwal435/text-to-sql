# app.py
"""
Main app using local RAG (embeddings.npy + corpus.json), new OpenAI client,
NearestNeighbors for retrieval, and Gradio UI.

Run:
1) python build_embeddings.py   # creates embeddings.npy + corpus.json
2) python app.py
Open http://localhost:7860
"""
import os
import re
import json
import sqlite3
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
import gradio as gr

load_dotenv()
client = OpenAI()  # uses OPENAI_API_KEY from .env

# Paths
EMB_PATH = "embeddings.npy"
CORPUS_PATH = "corpus.json"

# Retrieval settings
NN_K = 3
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"

# Safety regex (block destructive SQL)
DANGEROUS = re.compile(r"\b(DROP|DELETE|INSERT|UPDATE|TRUNCATE|ALTER|CREATE|REPLACE|GRANT|REVOKE|ATTACH|DETACH)\b", re.I)
SELECT_ONLY = re.compile(r"^\s*SELECT\b", re.I | re.DOTALL)

def is_safe_sql(sql: str):
    if DANGEROUS.search(sql):
        return False, "Forbidden keyword in SQL."
    if not SELECT_ONLY.search(sql.strip()):
        return False, "Only SELECT statements are allowed."
    inner = sql.strip().rstrip(";")
    if ";" in inner:
        return False, "Multiple statements are not allowed."
    return True, ""

# Load index
def load_index():
    if not os.path.exists(EMB_PATH) or not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError("embeddings.npy or corpus.json not found. Run build_embeddings.py first.")
    emb = np.load(EMB_PATH)
    with open(CORPUS_PATH, "r", encoding="utf8") as f:
        corpus = json.load(f)
    nn = NearestNeighbors(n_neighbors=NN_K, metric="cosine")
    nn.fit(emb)
    return nn, emb, corpus

nn, emb_matrix, corpus = load_index()

def embed_text(text: str):
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return np.array(resp.data[0].embedding, dtype="float32")

def retrieve(question: str, k=NN_K):
    q_emb = embed_text(question)
    distances, indices = nn.kneighbors([q_emb], n_neighbors=k)
    retrieved = [corpus[int(i)]["text"] for i in indices[0]]
    return retrieved

SYSTEM_INSTR = (
    "You are an assistant that converts user questions into a single valid SQL SELECT statement. "
    "Only output the SQL statement and nothing else. Use the CONTEXT if helpful. "
    "If ambiguous, provide a reasonable SELECT. Use MySQL/ANSI-style SQL."
)

def generate_sql(question: str, context: list[str]):
    # build messages
    message_system = SYSTEM_INSTR + "\n\nCONTEXT:\n" + "\n\n".join(context)
    messages = [
        {"role": "system", "content": message_system},
        {"role": "user", "content": question}
    ]
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=300
    )
    # Access content
    # New API returns object with choices list; message content at resp.choices[0].message.content
    content = resp.choices[0].message.content.strip()
    # clean up utility wrappers (remove code fences)
    content = content.strip("`\"' \n")
    if not content.endswith(";"):
        content = content + ";"
    return content

def try_repair(sql: str, error: str, context: list[str], question: str):
    prompt = (
        f"Original SQL:\n{sql}\n\nMySQL error:\n{error}\n\nContext:\n" + "\n\n".join(context) +
        "\n\nProvide a corrected SQL SELECT statement only. If you cannot fix without clarification, output: QUESTION: <your question>"
    )
    messages = [{"role": "user", "content": prompt}]
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=200
    )
    fixed = resp.choices[0].message.content.strip()
    fixed = fixed.strip("`\"' \n")
    if not fixed.endswith(";") and not fixed.startswith("QUESTION:"):
        fixed += ";"
    return fixed

# Gradio UI — this tool generates SQL variants and returns one SQL (agentic generate -> repair loop)
def agent_generate(question: str):
    # retrieve context
    context = retrieve(question, k=NN_K)
    # generate SQL
    sql = generate_sql(question, context)
    safe, reason = is_safe_sql(sql)
    if not safe:
        return sql, f"Blocked: {reason}", "\n\n".join(context)
    # Optionally we could execute; for this app we return SQL and context (execution requires DB)
    return sql, "Generated (no execution)", "\n\n".join(context)

# UI wrapper
def ui_generate(question: str):
    if not question or not question.strip():
        return "", "Please enter a natural language request.", ""
    try:
        sql, status, ctx = agent_generate(question)
        return sql, status, ctx
    except Exception as e:
        return "", f"Error: {e}", ""

with gr.Blocks(title="NL to SQL Generator") as demo:
    gr.Markdown(
        """# 🚀 Natural Language to SQL Generator
        Transform your questions into SQL queries instantly using AI-powered intelligence.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            q = gr.Textbox(
                label="💬 Ask your question in plain English",
                placeholder="e.g., Show me all students older than 20 with grade A",
                lines=3,
                max_lines=5
            )
            btn = gr.Button(
                "✨ Generate SQL",
                variant="primary",
                size="lg"
            )
        
        with gr.Column(scale=1):
            gr.Markdown(
                """### 💡 Example Questions:
                - List all student names and cities
                - Show students with grade A
                - Find students older than 25
                - Count students by grade
                """
            )
    
    with gr.Row():
        with gr.Column():
            sql_out = gr.Code(
                label="📝 Generated SQL Query",
                language="sql",
                lines=6
            )
            status_out = gr.Textbox(
                label="📊 Status",
                lines=1,
                interactive=False
            )
    
    with gr.Accordion("🔍 Retrieved Context (Advanced)", open=False):
        ctx_out = gr.Textbox(
            label="Context used for generation",
            lines=8,
            interactive=False
        )
    
    btn.click(fn=ui_generate, inputs=[q], outputs=[sql_out, status_out, ctx_out])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        share=False,
        show_error=True
    )
