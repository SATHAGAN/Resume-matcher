"""
Strict Pydantic v2 schemas for LLM structured extraction (Step 2 of the pipeline).

These are handed to the NIM chat model as a JSON schema in the system prompt so
the model has no ambiguity about the shape of the object it must return, and
they're used again on the way back in to validate/coerce the response.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class JDExtraction(BaseModel):
    title: str = Field(description="Job title as stated or best inferred from the JD")
    required_skills: List[str] = Field(default_factory=list, description="Hard requirements - must-have skills/tools/certs")
    preferred_skills: List[str] = Field(default_factory=list, description="Nice-to-have / bonus skills")
    experience_level: str = Field(description="e.g. Junior, Mid, Senior, Staff, Principal")
    min_years_experience: float = Field(default=0, description="Minimum years of experience required, numeric")
    domain: str = Field(description="Industry/domain, e.g. Fintech, Healthcare, E-commerce")

    @field_validator("required_skills", "preferred_skills", mode="before")
    @classmethod
    def _dedupe_lower(cls, v):
        if not v:
            return []
        seen, out = set(), []
        for item in v:
            key = str(item).strip()
            if key and key.lower() not in seen:
                seen.add(key.lower())
                out.append(key)
        return out


class ProjectEvidence(BaseModel):
    project_title: str = Field(description="Name of the project or role")
    architecture_used: str = Field(default="", description="System architecture / design pattern described")
    tech_stack: List[str] = Field(default_factory=list, description="Concrete technologies actually used in this project")
    scale_complexity: str = Field(default="", description="Scale indicators: users, requests/sec, data volume, team size, etc.")
    business_impact: str = Field(default="", description="Measurable outcome/impact produced by this project")


class ResumeExtraction(BaseModel):
    candidate_name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    total_experience_years: float = Field(default=0, description="Total professional experience in years, numeric")
    skills: List[str] = Field(default_factory=list, description="All skills/technologies mentioned anywhere in the resume")
    education: List[str] = Field(default_factory=list, description="Degrees, institutions, e.g. 'B.S. Computer Science - MIT'")
    certifications: List[str] = Field(default_factory=list)
    projects: List[ProjectEvidence] = Field(default_factory=list, description="Every distinct project/role with demonstrated evidence")

    @field_validator("email", mode="before")
    @classmethod
    def _blank_email_to_none(cls, v):
        return v or None


class XAIExplanation(BaseModel):
    strengths: List[str] = Field(default_factory=list, description="2-5 concrete reasons this candidate scored well")
    skill_gaps: List[str] = Field(default_factory=list, description="You are an expert technical recruiter analyzing a resume against a job description. When extracting skill_gaps, identify 1 to 3 specific, critical skills missing from the resume and format them as short, human-readable phrases (e.g., Missing experience with CI/CD pipelines). You must strictly avoid copying and pasting raw, mashed-together keywords from the JD (never output -JavaAWSMySQLDocker). Return the gaps as a clean list of distinct, properly spaced strings, or output [No critical gaps identified] if the candidate meets all major requirements.")
    reasoning: str = Field(description="2-4 sentence plain-English explanation of the overall score")
    suggested_interview_questions: List[str] = Field(
        default_factory=list, description="3-5 questions probing this candidate's specific projects"
    )
