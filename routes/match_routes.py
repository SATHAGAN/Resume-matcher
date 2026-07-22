from flask import Blueprint, render_template, redirect, url_for, flash, current_app

from database import db
from database.models import JobDescription, Resume, MatchResult
from services import vector_service, scoring_engine, xai_service, nim_client

match_bp = Blueprint("match", __name__)


@match_bp.route("/")
def home():
    jds = JobDescription.query.order_by(JobDescription.created_at.desc()).all()
    resume_count = Resume.query.count()
    return render_template("home.html", jds=jds, resume_count=resume_count)


@match_bp.route("/match/<int:jd_id>")
def dashboard(jd_id):
    jd = JobDescription.query.get_or_404(jd_id)
    resume_count = Resume.query.count()

    results = (
        MatchResult.query
        .filter_by(job_description_id=jd_id)
        .order_by(MatchResult.final_score.desc())
        .all()
    )
    # Split purely on whether the XAI writeup succeeded, not on position - if an XAI
    # call fails partway through the top N, that candidate falls through to the plain
    # table instead of being duplicated or silently dropped.
    top_results = [r for r in results if r.xai_summary]
    other_results = [r for r in results if not r.xai_summary]

    return render_template(
        "dashboard.html",
        jd=jd,
        resume_count=resume_count,
        top_results=top_results,
        other_results=other_results,
        has_results=bool(results),
    )


@match_bp.route("/match/<int:jd_id>/run", methods=["POST"])
def run_match(jd_id):
    """
    Steps 4-6, triggered on demand ('Find Candidates'):
      4. hybrid search -> top 50 resumes
      5. deterministic multi-factor scoring for all 50
      6. XAI writeup for the top 10 only
    """
    jd = JobDescription.query.get_or_404(jd_id)

    if Resume.query.count() == 0:
        flash("Upload at least one resume before running a match.", "error")
        return redirect(url_for("match.dashboard", jd_id=jd_id))

    try:
        # Step 4: hybrid search retrieval (cheap - no LLM call)
        candidates = vector_service.hybrid_search(jd, top_k=current_app.config["HYBRID_SEARCH_TOP_K"])

        # Step 5: deterministic scoring for every retrieved candidate
        scored = []
        for resume, _hybrid_score, vector_similarity in candidates:
            breakdown = scoring_engine.compute_score(jd, resume, resume.projects, vector_similarity)
            scored.append((resume, breakdown))
        scored.sort(key=lambda t: t[1]["final_score"], reverse=True)

        # Clear any previous run for this JD so re-running reflects the current resume bank.
        MatchResult.query.filter_by(job_description_id=jd.id).delete()

        top_n = current_app.config["XAI_TOP_N"]
        for i, (resume, breakdown) in enumerate(scored):
            xai_summary = None
            if i < top_n:
                try:
                    # Step 6: XAI writeup - only for the top N
                    xai_summary = xai_service.generate_explanation(jd, resume, resume.projects, breakdown)
                except nim_client.NIMError as exc:
                    flash(f"XAI writeup failed for {resume.candidate_name}: {exc}", "error")

            db.session.add(MatchResult(
                job_description_id=jd.id,
                resume_id=resume.id,
                final_score=breakdown["final_score"],
                score_breakdown=breakdown,
                xai_summary=xai_summary,
            ))

        db.session.commit()
        flash(f"Matching complete - scored {len(scored)} candidates.", "success")

    except nim_client.NIMError as exc:
        db.session.rollback()
        flash(f"Matching failed: {exc}", "error")
    except Exception as exc:  # noqa: BLE001 - surface any unexpected pipeline error to the recruiter
        db.session.rollback()
        flash(f"Matching failed unexpectedly: {exc}", "error")

    return redirect(url_for("match.dashboard", jd_id=jd_id))
