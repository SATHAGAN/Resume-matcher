from flask import Blueprint, render_template, request, redirect, url_for, flash, current_app

from database import db
from database.models import JobDescription, Resume, CandidateProject
import services.text_extractor as text_extractor
import services.ai_extractor as ai_extractor
import services.vector_service as vector_service
import services.nim_client as nim_client

upload_bp = Blueprint("upload", __name__)


def _allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in current_app.config["UPLOAD_ALLOWED_EXTENSIONS"]


# ---------------------------------------------------------------- Job Descriptions

@upload_bp.route("/jd/new", methods=["GET", "POST"])
def new_jd():
    if request.method == "GET":
        return render_template("new_jd.html")

    pasted_text = (request.form.get("raw_text") or "").strip()
    uploaded_file = request.files.get("jd_file")

    try:
        if uploaded_file and uploaded_file.filename:
            if not _allowed_file(uploaded_file.filename):
                flash("Only .pdf and .docx files are supported.", "error")
                return redirect(url_for("upload.new_jd"))
            raw_text = text_extractor.extract_text(uploaded_file)
        elif pasted_text:
            raw_text = pasted_text
        else:
            flash("Paste the job description text or upload a file.", "error")
            return redirect(url_for("upload.new_jd"))

        extraction = ai_extractor.extract_jd(raw_text)
        embedding = vector_service.embed_jd_text(extraction, raw_text)

        jd = JobDescription(
            title=extraction.title,
            raw_text=raw_text,
            required_skills=extraction.required_skills,
            preferred_skills=extraction.preferred_skills,
            experience_level=extraction.experience_level,
            min_years_experience=extraction.min_years_experience,
            domain=extraction.domain,
            embedding=embedding,
        )
        db.session.add(jd)
        db.session.commit()

        flash(f"Job description \u201c{jd.title}\u201d added.", "success")
        
        # Pass the newly created JD ID to the resume upload route
        return redirect(url_for("upload.new_resume", jd_id=jd.id))

    except nim_client.NIMError as exc:
        flash(f"AI extraction failed: {exc}", "error")
        return redirect(url_for("upload.new_jd"))
    except Exception as exc:  
        flash(f"Could not process job description: {exc}", "error")
        return redirect(url_for("upload.new_jd"))


# ---------------------------------------------------------------- Resumes

@upload_bp.route("/resumes/new", methods=["GET", "POST"])
def new_resume():
    # Grab the Job Description ID from the URL
    jd_id = request.args.get("jd_id")
    
    if request.method == "GET":
        # Pass the jd_id to the template so the form action keeps it
        return render_template("new_resume.html", jd_id=jd_id)

    files = [f for f in request.files.getlist("resume_files") if f and f.filename]
    if not files:
        flash("Select at least one .pdf or .docx resume to upload.", "error")
        return redirect(url_for("upload.new_resume", jd_id=jd_id))

    added, errors = [], []

    for f in files:
        try:
            if not _allowed_file(f.filename):
                errors.append(f"{f.filename}: unsupported file type (use .pdf or .docx)")
                continue

            raw_text = text_extractor.extract_text(f)
            extraction = ai_extractor.extract_resume(raw_text)
            embedding = vector_service.embed_resume_text(extraction, raw_text)

            resume = Resume(
                job_description_id=jd_id,  # Locks this resume to the specific job
                candidate_name=extraction.candidate_name,
                email=extraction.email,
                phone=extraction.phone,
                raw_text=raw_text,
                total_experience_years=extraction.total_experience_years,
                skills=extraction.skills,
                education=extraction.education,
                certifications=extraction.certifications,
                embedding=embedding,
            )
            db.session.add(resume)
            db.session.flush() 

            for proj in extraction.projects:
                db.session.add(CandidateProject(
                    resume_id=resume.id,
                    title=proj.project_title,
                    architecture=proj.architecture_used,
                    tech_stack=proj.tech_stack,
                    scale_complexity=proj.scale_complexity,
                    business_impact=proj.business_impact,
                ))

            db.session.commit()
            vector_service.index_resume(resume)  
            added.append(resume.candidate_name)

        except nim_client.NIMError as exc:
            db.session.rollback()
            errors.append(f"{f.filename}: AI extraction failed ({exc})")
        except Exception as exc:  
            db.session.rollback()
            errors.append(f"{f.filename}: {exc}")

    if added:
        flash(f"Added {len(added)} candidate(s): {', '.join(added)}", "success")
    for err in errors:
        flash(err, "error")

    # Redirect straight to the match dashboard for this specific job
    return redirect(url_for("match.dashboard", jd_id=jd_id))