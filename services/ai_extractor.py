"""
Step 2: LLM Structured Extraction.

Sends raw JD/resume text to the NIM chat model and forces it into our strict
Pydantic schemas. We hand the model the actual JSON schema (via
model_json_schema()) so there's no ambiguity about field names/types, then
validate the response and retry once with the validation error fed back in
if it doesn't conform.
"""
from services import nim_client
from schemas.extraction_schema import JDExtraction, ResumeExtraction

_SYSTEM_PREAMBLE = (
    "You are a precise information-extraction engine for a recruiting platform. "
    "You extract ONLY facts that are explicitly present or very clearly implied in the "
    "supplied text. Never invent skills, employers, dates, or numbers. "
    "Respond with a single JSON object and nothing else - no markdown, no commentary."
)


def extract_jd(raw_text: str) -> JDExtraction:
    return nim_client.run_structured_extraction(
        raw_text,
        JDExtraction,
        "Extract structured fields from this job description: required vs. preferred "
        "skills, seniority level, minimum years of experience, and domain/industry.",
        _SYSTEM_PREAMBLE,
    )


def extract_resume(raw_text: str) -> ResumeExtraction:
    return nim_client.run_structured_extraction(
        raw_text,
        ResumeExtraction,
        "Extract structured fields from this resume. Pay special attention to the "
        "`projects` array: identify every distinct project or role, and for each one "
        "capture the architecture used, the concrete tech stack, scale/complexity "
        "indicators (users, throughput, data volume, team size), and measurable "
        "business impact. Only include a project if the resume actually describes what "
        "was built/done, not just a job title.",
        _SYSTEM_PREAMBLE,
    )
