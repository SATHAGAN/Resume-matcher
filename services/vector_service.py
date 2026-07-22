"""
Step 3 (Vectorization) and Step 4 (Hybrid Search Match Trigger).

There's no pgvector here since we're on SQLite, so:
  - Embeddings (1024-dim, from NIM's nvidia/nv-embedqa-e5-v5) are stored as a
    JSON float array on each row and compared with plain numpy cosine
    similarity. Fine at hundreds-to-low-thousands of resumes; if this needs
    to scale further, swap this module for pgvector + HNSW without touching
    any calling code.
  - Keyword search uses a real SQLite FTS5 virtual table + bm25() ranking,
    kept in sync manually on resume create/update (see index_resume below).
"""
import sqlite3
import numpy as np
from flask import current_app
from database import db
from database.models import Resume
from services import nim_client


# ---------- embeddings ----------

def embed_jd_text(jd_extraction, raw_text) -> list:
    summary = (
        f"Job Title: {jd_extraction.title}\n"
        f"Domain: {jd_extraction.domain}\n"
        f"Seniority: {jd_extraction.experience_level}\n"
        f"Required skills: {', '.join(jd_extraction.required_skills)}\n"
        f"Preferred skills: {', '.join(jd_extraction.preferred_skills)}\n\n"
        f"{raw_text}"
    )
    return nim_client.get_embedding(summary, input_type="query")


def embed_resume_text(resume_extraction, raw_text) -> list:
    summary = (
        f"Candidate: {resume_extraction.candidate_name}\n"
        f"Skills: {', '.join(resume_extraction.skills)}\n"
        f"Experience: {resume_extraction.total_experience_years} years\n\n"
        f"{raw_text}"
    )
    return nim_client.get_embedding(summary, input_type="passage")


def cosine_similarity(vec_a, vec_b) -> float:
    """Returns cosine similarity rescaled from [-1, 1] to [0, 1]."""
    if not vec_a or not vec_b:
        return 0.0
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    sim = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, (sim + 1) / 2))


# ---------- FTS5 keyword index ----------

def init_fts_table():
    """Call once at app startup (after db.create_all())."""
    conn = db.engine.raw_connection()
    try:
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS resume_fts USING fts5(
                resume_id UNINDEXED,
                searchable_text
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def index_resume(resume: Resume):
    """(Re)index a single resume's keyword-searchable text."""
    searchable = " ".join([
        resume.candidate_name or "",
        " ".join(resume.skills or []),
        " ".join(resume.certifications or []),
        resume.raw_text or "",
    ])
    conn = db.engine.raw_connection()
    try:
        conn.execute("DELETE FROM resume_fts WHERE resume_id = ?", (resume.id,))
        conn.execute(
            "INSERT INTO resume_fts (resume_id, searchable_text) VALUES (?, ?)",
            (resume.id, searchable),
        )
        conn.commit()
    finally:
        conn.close()


def _keyword_scores(jd) -> dict:
    """Returns {resume_id: normalized_bm25_score_0_to_1} via FTS5 MATCH."""
    terms = list(jd.required_skills or []) + list(jd.preferred_skills or [])
    terms = [t.strip() for t in terms if t and t.strip()]
    if not terms:
        return {}

    # Quote each term so multi-word skills ("machine learning") stay as phrases, OR'd together.
    match_query = " OR ".join(f'"{t}"' for t in terms)

    conn = db.engine.raw_connection()
    try:
        cur = conn.execute(
            """
            SELECT resume_id, bm25(resume_fts) AS rank
            FROM resume_fts
            WHERE resume_fts MATCH ?
            ORDER BY rank
            """,
            (match_query,),
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        return {}  # a skill term collided with FTS5 query syntax - fail soft, vector search still runs
    finally:
        conn.close()

    if not rows:
        return {}

    # SQLite's bm25(): more negative = better match. Min-max normalize -> invert into 0..1 "goodness".
    ranks = [r[1] for r in rows]
    best, worst = min(ranks), max(ranks)
    spread = (worst - best) or 1.0
    return {resume_id: 1.0 - ((rank - best) / spread) for resume_id, rank in rows}

def hybrid_search(jd, top_k=None):
    """
    Retrieves the top_k resumes for a JD by blending keyword (BM25) and
    semantic (cosine) scores - without running the LLM against every resume
    in the bank.
    
    ISOLATION UPDATE: Only searches resumes explicitly linked to this JD.
    """
    top_k = top_k or current_app.config["HYBRID_SEARCH_TOP_K"]
    keyword_scores = _keyword_scores(jd)

    scored = []
    
    # NEW: Filter by the current job_description_id instead of querying all resumes
    for resume in Resume.query.filter_by(job_description_id=jd.id).all():
        vec_score = cosine_similarity(jd.embedding, resume.embedding)
        kw_score = keyword_scores.get(resume.id, 0.0)
        hybrid = 0.5 * kw_score + 0.5 * vec_score
        scored.append((resume, hybrid, vec_score))

    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:top_k]