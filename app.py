import os
from flask import Flask

from config import Config
from database import db


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_UPLOAD_MB"] * 1024 * 1024

    os.makedirs(os.path.join(os.path.dirname(__file__), "instance"), exist_ok=True)

    db.init_app(app)

    from routes.upload_routes import upload_bp
    from routes.match_routes import match_bp
    app.register_blueprint(upload_bp)
    app.register_blueprint(match_bp)

    with app.app_context():
        db.create_all()
        from services import vector_service
        vector_service.init_fts_table()

    @app.context_processor
    def inject_factor_labels():
        return {
            "FACTOR_LABELS": {
                "skill_match": "Skill match",
                "skill_evidence_in_projects": "Skill evidence in projects",
                "project_complexity": "Project complexity",
                "experience_years": "Experience (years)",
                "semantic_vector_similarity": "Semantic similarity",
                "education_certs": "Education / certs",
            }
        }

    @app.template_filter("score_tier")
    def score_tier(score):
        """Maps a 0-100 score to a CSS class for the evidence-ledger styling."""
        if score is None:
            return "tier-unknown"
        if score >= 75:
            return "tier-strong"
        if score >= 50:
            return "tier-moderate"
        return "tier-weak"

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
