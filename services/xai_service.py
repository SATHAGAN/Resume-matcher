"""
Step 6: Explainable AI (XAI) Dashboard Output.

Only the top N (default 10) already-scored candidates get this call - the
scoring engine (deterministic, cheap) does the heavy lifting of narrowing
down the field first.
"""
from schemas.extraction_schema import XAIExplanation
from services import nim_client

_SYSTEM_PREAMBLE = (
    "You are an experienced technical recruiter writing an audit of an automated "
    "candidate score. Be specific and evidence-based; never invent facts not present "
    "in the context you're given. Respond with a single JSON object and nothing else."
)


def generate_explanation(jd, resume, projects, score_breakdown) -> dict:
    projects_desc = "\n".join(
        f"- {p.title}: architecture={p.architecture!r}, tech_stack={p.tech_stack}, "
        f"scale={p.scale_complexity!r}, impact={p.business_impact!r}"
        for p in projects
    ) or "(no distinct projects extracted)"

    context = f"""
JOB DESCRIPTION
Title: {jd.title}
Domain: {jd.domain}
Seniority: {jd.experience_level}
Required skills: {jd.required_skills}
Preferred skills: {jd.preferred_skills}

CANDIDATE
Name: {resume.candidate_name}
Total experience: {resume.total_experience_years} years
Skills listed: {resume.skills}
Education: {resume.education}
Certifications: {resume.certifications}

DEMONSTRATED PROJECT EVIDENCE
{projects_desc}

SCORING ENGINE OUTPUT (0..1 per factor, deterministic, already computed - do not recompute)
{score_breakdown['sub_scores']}
Final score (0..100): {score_breakdown['final_score']}
""".strip()

    result = nim_client.run_structured_extraction(
        context,
        XAIExplanation,
        "Given the JD, the candidate's extracted profile/project evidence, and the "
        "already-computed score breakdown above, explain the score in plain English. "
        "Call out specific strengths backed by named projects, name any required/"
        "preferred skills that are missing or unevidenced, and write interview "
        "questions that probe the candidate's own projects specifically (not generic "
        "questions).",
        _SYSTEM_PREAMBLE,
    )
    return result.model_dump()
