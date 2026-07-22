"""
Step 5: The Multi-Factor Scoring Engine.

Pure Python, fully deterministic - no LLM call in this module. Runs against
the top_k candidates returned by vector_service.hybrid_search.

    Final_Score = 0.30 * Skill_Match
                + 0.20 * Skill_Evidence_in_Projects
                + 0.15 * Project_Complexity
                + 0.15 * Experience_Years
                + 0.10 * Semantic_Vector_Similarity
                + 0.10 * Education_Certs

Every sub-score is normalized to 0..1 before weighting; final_score is
reported on a 0..100 scale.
"""

WEIGHTS = {
    "skill_match": 0.15,
    "skill_evidence_in_projects": 0.30,
    "project_complexity": 0.25,
    "experience_years": 0.05,
    "semantic_vector_similarity": 0.20,
    "education_certs": 0.05,
}

_LEVEL_TO_YEARS = {
    "intern": 0, "junior": 1, "entry": 1, "mid": 3, "mid-level": 3,
    "senior": 5, "staff": 8, "principal": 10, "lead": 7, "director": 12,
}

_COMPLEXITY_SIGNALS = [
    # Backend / Data
    "million", "billion", "microservice", "distributed", "high availability",
    "scalab", "concurrent", "real-time", "realtime", "kubernetes", "load balanc",
    "sharding", "petabyte", "terabyte", "sla", "99.9", "throughput", "latency",
    "fault-toleran", "multi-region", "high-throughput", "low-latency", "replication",
    
    # Frontend / UI / Mobile
    "accessibility", "a11y", "state management", "responsive", "web vitals", 
    "cross-platform", "figma", "design system", "component library",
    
    # General Impact / Process
    "conversion rate", "seo", "ci/cd", "pipeline", "optimization", "refactor"
]


def _norm(items):
    return {str(x).strip().lower() for x in (items or []) if str(x).strip()}


def _skill_match(jd, resume) -> float:
    required = _norm(jd.required_skills)
    preferred = _norm(jd.preferred_skills)
    candidate = _norm(resume.skills)

    req_score = (len(required & candidate) / len(required)) if required else 1.0
    pref_score = (len(preferred & candidate) / len(preferred)) if preferred else 1.0
    return 0.7 * req_score + 0.3 * pref_score


def _skill_evidence_in_projects(jd, projects) -> float:
    required = _norm(jd.required_skills)
    if not required:
        return 1.0 if projects else 0.0

    evidence_blob = " ".join(
        " ".join([
            p.title or "", p.architecture or "", p.business_impact or "",
            " ".join(p.tech_stack or []),
        ]).lower()
        for p in projects
    )
    if not evidence_blob.strip():
        return 0.0

    found = sum(1 for skill in required if skill in evidence_blob)
    return found / len(required)


def _project_complexity(projects) -> float:
    if not projects:
        return 0.0

    blob = " ".join(
        " ".join([p.architecture or "", p.scale_complexity or ""]).lower()
        for p in projects
    )
    signal_hits = sum(1 for sig in _COMPLEXITY_SIGNALS if sig in blob)
    signal_score = min(1.0, signal_hits / 5)
    breadth_score = min(1.0, len(projects) / 3)
    return 0.7 * signal_score + 0.3 * breadth_score


def _experience_years(jd, resume) -> float:
    required_years = jd.min_years_experience or _LEVEL_TO_YEARS.get(
        (jd.experience_level or "").strip().lower(), 3
    )
    if required_years <= 0:
        return 1.0
    return min(1.0, (resume.total_experience_years or 0) / required_years)


def _education_certs(jd, resume) -> float:
    degree_signals = ("bachelor", "master", "phd", "b.s", "b.sc", "m.s", "m.sc", "b.tech", "m.tech", "bs ", "ms ")
    education_blob = " ".join(resume.education or []).lower()
    has_degree = any(sig in education_blob for sig in degree_signals)

    cert_score = min(1.0, len(resume.certifications or []) / 2)
    return 0.5 * (1.0 if has_degree else 0.0) + 0.5 * cert_score


def compute_score(jd, resume, projects, vector_similarity: float) -> dict:
    """
    jd: JobDescription ORM instance
    resume: Resume ORM instance
    projects: list of CandidateProject ORM instances for that resume
    vector_similarity: precomputed cosine similarity from hybrid_search (0..1)

    Returns a dict with every sub-score (0..1), the weights used, and
    final_score on a 0..100 scale.
    """
    sub_scores = {
        "skill_match": _skill_match(jd, resume),
        "skill_evidence_in_projects": _skill_evidence_in_projects(jd, projects),
        "project_complexity": _project_complexity(projects),
        "experience_years": _experience_years(jd, resume),
        "semantic_vector_similarity": max(0.0, min(1.0, vector_similarity)),
        "education_certs": _education_certs(jd, resume),
    }

    weighted_total = sum(sub_scores[k] * WEIGHTS[k] for k in WEIGHTS)

    return {
        "sub_scores": {k: round(v, 3) for k, v in sub_scores.items()},
        "weights": WEIGHTS,
        "final_score": round(weighted_total * 100, 1),
    }
