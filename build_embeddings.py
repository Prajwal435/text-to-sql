# build_embeddings.py
"""
Build local embeddings (embeddings.npy) and corpus.json using the new OpenAI client.
Run: python build_embeddings.py
"""
import os
import json
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()  # reads OPENAI_API_KEY from env

# Small corpus: schema + few-shot examples
corpus = [
    {
        "id": "schema_student",
        "text": (
            "Table: student\n"
            "Columns:\n"
            "  - student_id (INT PRIMARY KEY)\n"
            "  - name (VARCHAR)\n"
            "  - age (INT)\n"
            "  - grade (VARCHAR)\n"
            "  - city (VARCHAR)\n"
            "Description: basic student table with id, name, age, grade and city."
        )
    },
    {"id": "ex1", "text": "NL: List all student names and their cities.\nSQL: SELECT name, city FROM student;"},
    {"id": "ex2", "text": "NL: Show students older than 20.\nSQL: SELECT * FROM student WHERE age > 20;"},
    {"id": "ex3", "text": "NL: Get names of students with grade A.\nSQL: SELECT name FROM student WHERE grade = 'A';"}
]

EMBEDDING_MODEL = "text-embedding-3-small"  # change if desired

def get_embedding(text: str):
    """
    Use OpenAI client (new API) to create an embedding.
    Returns a float list (embedding vector).
    """
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    # response structure: resp.data[0].embedding
    return resp.data[0].embedding

def build():
    texts = [c["text"] for c in corpus]
    ids = [c["id"] for c in corpus]
    embeddings = []
    for t in texts:
        print("Embedding:", t.splitlines()[0][:80])
        e = get_embedding(t)
        embeddings.append(e)
    embs = np.array(embeddings, dtype="float32")
    np.save("embeddings.npy", embs)
    with open("corpus.json", "w", encoding="utf8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
    print("Saved embeddings.npy and corpus.json ({} vectors)".format(len(embs)))

if __name__ == "__main__":
    build()
