from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app
from database import db
from database.models import JobDescription, Resume, MatchResult, CandidateProject
import services.vector_service as vector_service
import services.scoring_engine as scoring_engine
import services.xai_service as xai_service

match_bp = Blueprint("match", __name__)

@match_bp.route("/")
def home():
    jobs = JobDescription.query.order_by(JobDescription.created_at.desc()).all()
    
    # Count resumes per job dynamically
    job_data = []
    for jd in jobs:
        r_count = Resume.query.filter_by(job_description_id=jd.id).count()
        job_data.append({"jd": jd, "resume_count": r_count})
        
    # FIX: Pass both 'jobs' and 'job_data' so home.html can read them
    return render_template("home.html", jobs=jobs, job_data=job_data)



@match_bp.route("/clear-all", methods=["POST"])
def clear_all():
    """Wipes all jobs, resumes, and match history from the database."""
    try:
        MatchResult.query.delete()
        CandidateProject.query.delete()
        Resume.query.delete()
        JobDescription.query.delete()
        db.session.commit()
        flash("All workspace data has been successfully cleared.", "success")
    except Exception as exc:
        db.session.rollback()
        flash(f"Failed to clear data: {exc}", "error")
    return redirect(url_for("match.home"))


@match_bp.route("/jd/<int:jd_id>/match", methods=["POST"])
def run_match(jd_id):
    """
    Computes or updates match scores and XAI summaries for all resumes 
    linked to this specific Job Description.
    """
    jd = JobDescription.query.get_or_404(jd_id)
    resumes = Resume.query.filter_by(job_description_id=jd_id).all()

    if not resumes:
        flash("No resumes found for this job description to analyze.", "error")
        return redirect(url_for("match.dashboard", jd_id=jd_id))

    # 1. Run hybrid search to rank candidates by vector/keyword relevance
    candidates = vector_service.hybrid_search(jd, top_k=len(resumes))

    for resume, hybrid_score, vector_similarity in candidates:
        # 2. Compute deterministic sub-scores
        breakdown = scoring_engine.compute_score(jd, resume, resume.projects, vector_similarity)

        # 3. Check if a match result already exists for this resume & job
        existing_result = MatchResult.query.filter_by(
            job_description_id=jd.id, resume_id=resume.id
        ).first()

        if existing_result:
            # UPDATE existing score and breakdown so ranks update correctly
            existing_result.final_score = breakdown["final_score"]
            existing_result.score_breakdown = breakdown
        else:
            # CREATE new result if it's a newly added resume
            new_result = MatchResult(
                job_description_id=jd.id,
                resume_id=resume.id,
                final_score=breakdown["final_score"],
                score_breakdown=breakdown,
            )
            db.session.add(new_result)

    db.session.commit()

    # 4. Generate XAI write-ups for the top candidates
    top_n = current_app.config["XAI_TOP_N"]
    top_results = MatchResult.query.filter_by(job_description_id=jd_id)\
        .order_by(MatchResult.final_score.desc())\
        .limit(top_n).all()

    for r in top_results:
        if not r.xai_summary:  # Generate if missing
            try:
                r.xai_summary = xai_service.generate_xai(jd, r.resume, r.score_breakdown)
                db.session.commit()
            except Exception:
                db.session.rollback()

    flash("Candidate rankings and scores successfully updated!", "success")
    return redirect(url_for("match.dashboard", jd_id=jd_id))


@match_bp.route("/jd/<int:jd_id>/dashboard")
def dashboard(jd_id):
    jd = JobDescription.query.get_or_404(jd_id)
    
    resume_count = Resume.query.filter_by(job_description_id=jd_id).count()
    results = MatchResult.query.filter_by(job_description_id=jd_id)\
        .order_by(MatchResult.final_score.desc())\
        .all()

    has_results = len(results) > 0
    
    # SMART AUTO-UPDATE: If new resumes were added that haven't been scored yet, 
    # automatically trigger a background match calculation!
    if resume_count > len(results) and resume_count > 0:
        return run_match(jd_id)

    top_n = current_app.config["XAI_TOP_N"]
    top_results = results[:top_n]
    other_results = results[top_n:]

    return render_template(
        "dashboard.html",
        jd=jd,
        resume_count=resume_count,
        has_results=has_results,
        top_results=top_results,
        other_results=other_results,
    )
