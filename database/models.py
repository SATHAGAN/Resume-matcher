from datetime import datetime, timezone
from database import db


def utcnow():
    return datetime.now(timezone.utc)


class JobDescription(db.Model):
    __tablename__ = "job_descriptions"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    raw_text = db.Column(db.Text, nullable=False)

    required_skills = db.Column(db.JSON, default=list)
    preferred_skills = db.Column(db.JSON, default=list)
    experience_level = db.Column(db.String(64))
    min_years_experience = db.Column(db.Float, default=0)
    domain = db.Column(db.String(128))

    # 1024-dim NIM embedding (nv-embedqa-e5-v5, input_type="query"), stored as JSON float list
    embedding = db.Column(db.JSON)

    created_at = db.Column(db.DateTime, default=utcnow)

    match_results = db.relationship(
        "MatchResult", backref="job_description", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<JobDescription {self.id} {self.title!r}>"


class Resume(db.Model):
    __tablename__ = "resumes"

    id = db.Column(db.Integer, primary_key=True)
    candidate_name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255))
    phone = db.Column(db.String(64))
    raw_text = db.Column(db.Text, nullable=False)

    total_experience_years = db.Column(db.Float, default=0)
    skills = db.Column(db.JSON, default=list)
    education = db.Column(db.JSON, default=list)
    certifications = db.Column(db.JSON, default=list)

    # 1024-dim NIM embedding (nv-embedqa-e5-v5, input_type="passage"), stored as JSON float list
    embedding = db.Column(db.JSON)

    created_at = db.Column(db.DateTime, default=utcnow)

    projects = db.relationship(
        "CandidateProject", backref="resume", cascade="all, delete-orphan"
    )
    match_results = db.relationship(
        "MatchResult", backref="resume", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Resume {self.id} {self.candidate_name!r}>"


class CandidateProject(db.Model):
    """Demonstrated skill evidence extracted from a resume's project/work history."""

    __tablename__ = "candidate_projects"

    id = db.Column(db.Integer, primary_key=True)
    resume_id = db.Column(db.Integer, db.ForeignKey("resumes.id"), nullable=False)

    title = db.Column(db.String(255))
    architecture = db.Column(db.Text)
    tech_stack = db.Column(db.JSON, default=list)
    scale_complexity = db.Column(db.Text)
    business_impact = db.Column(db.Text)

    def __repr__(self):
        return f"<CandidateProject {self.id} {self.title!r}>"


class MatchResult(db.Model):
    """Cached output of the scoring engine + XAI writeup for one (JD, Resume) pair."""

    __tablename__ = "match_results"

    id = db.Column(db.Integer, primary_key=True)
    job_description_id = db.Column(db.Integer, db.ForeignKey("job_descriptions.id"), nullable=False)
    resume_id = db.Column(db.Integer, db.ForeignKey("resumes.id"), nullable=False)

    final_score = db.Column(db.Float, nullable=False)
    score_breakdown = db.Column(db.JSON)     # per-factor scores from scoring_engine
    xai_summary = db.Column(db.JSON)         # strengths / gaps / interview questions

    created_at = db.Column(db.DateTime, default=utcnow)

    __table_args__ = (
        db.UniqueConstraint("job_description_id", "resume_id", name="uq_jd_resume"),
    )
