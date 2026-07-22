# Evidence Desk — AI Resume Matcher (Flask + NVIDIA NIM)

A Flask rebuild of the enterprise resume-matching pipeline, using **NVIDIA NIM**
for both LLM extraction and embeddings, **SQLite** for storage, and server-rendered
**Jinja2 HTML templates** (no React/build step).

## Pipeline (matches the original 6-step spec)

1. **Ingestion** — upload a JD and resumes as `.pdf`/`.docx` (or paste JD text). Text is
   pulled with PyMuPDF / python-docx.
2. **LLM structured extraction** — raw text is sent to a NIM chat model, forced into strict
   Pydantic v2 schemas (`schemas/extraction_schema.py`). Resumes get a detailed `projects`
   array capturing architecture, tech stack, scale, and business impact per project.
3. **Vectorization** — JD/resume summaries are embedded with NIM's `nvidia/nv-embedqa-e5-v5`
   (1024-dim, asymmetric query/passage model) and stored as JSON arrays on each row.
4. **Hybrid search** — clicking "Find candidates" runs a SQLite FTS5 BM25 keyword search
   blended with cosine vector similarity to pull the top 50 resumes, *without* calling the
   LLM against the whole bank.
5. **Multi-factor scoring** — a pure Python, fully deterministic formula
   (`services/scoring_engine.py`) scores all 50:
   `0.30·SkillMatch + 0.20·SkillEvidenceInProjects + 0.15·ProjectComplexity + 0.15·ExperienceYears + 0.10·SemanticSimilarity + 0.10·EducationCerts`
6. **XAI writeup** — the top 10 scored candidates get one more NIM call for a plain-English
   explanation, skill gaps, and tailored interview questions, rendered on the dashboard.

## Why SQLite instead of pgvector

This uses SQLite + FTS5 + numpy cosine similarity instead of Postgres/pgvector so it runs
with zero external services. It's a straight swap: `services/vector_service.py` is the only
file that would need to change to move to pgvector + HNSW at real scale — nothing else in
the app depends on the storage engine.

## Setup

```bash
cd resume_matcher
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set NVIDIA_API_KEY (get a free key at https://build.nvidia.com)

python app.py
```

Visit `http://localhost:5000`.

## Getting an NVIDIA NIM API key

1. Sign in at [build.nvidia.com](https://build.nvidia.com) (free).
2. Open any model card (e.g. `meta/llama-3.3-70b-instruct`) and click **Get API Key**.
3. Copy the key (starts with `nvapi-`) into `.env` as `NVIDIA_API_KEY`.

Swap `NVIDIA_CHAT_MODEL` in `.env` for any other instruct model in the NIM catalog if you'd
prefer — the extraction/XAI code doesn't assume a specific one, it just needs a model that's
reasonably good at following a JSON schema.

## Project layout

```
resume_matcher/
├── app.py                      # Flask app factory
├── config.py                   # env-driven config
├── database/
│   └── models.py               # JobDescription, Resume, CandidateProject, MatchResult
├── schemas/
│   └── extraction_schema.py    # Pydantic v2 schemas the LLM must conform to
├── services/
│   ├── nim_client.py           # NIM chat + embeddings wrapper, schema-validated extraction
│   ├── text_extractor.py       # PDF/DOCX -> raw text
│   ├── ai_extractor.py         # Step 2
│   ├── vector_service.py       # Steps 3-4 (embeddings, FTS5 + cosine hybrid search)
│   ├── scoring_engine.py       # Step 5 (deterministic formula)
│   └── xai_service.py          # Step 6
├── routes/
│   ├── upload_routes.py        # /jd/new, /resumes/new
│   └── match_routes.py         # /, /match/<id>, /match/<id>/run
└── templates/                  # Jinja2 HTML (base, home, forms, dashboard)
```

## Notes / limitations

- Vector search loads all resumes into memory and computes cosine similarity in Python —
  fine to a few thousand resumes; swap `vector_service.py` for pgvector to scale further.
- NIM's free tier is rate-limited (~40 req/min per the catalog docs); uploading many resumes
  back-to-back may need a short pause between batches.
- Environment here has no outbound network access, so this was validated with syntax
  checks (`py_compile`) and a template-rendering harness against mock data, not a live NIM
  call — sanity-check the two NIM endpoints against your own key before relying on it.
