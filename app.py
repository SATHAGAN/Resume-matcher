import os
from flask import Flask

from config import Config
from database import db

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Ensure MAX_UPLOAD_MB is defined in your config, otherwise default to a safe value
    max_mb = app.config.get("MAX_UPLOAD_MB", 16) 
    app.config["MAX_CONTENT_LENGTH"] = max_mb * 1024 * 1024

    # Create instance folder for local dev/storage
    os.makedirs(os.path.join(os.path.dirname(__file__), "instance"), exist_ok=True)

    db.init_app(app)

    # Register Blueprints
    from routes.upload_routes import upload_bp
    from routes.match_routes import match_bp
    app.register_blueprint(upload_bp)
    app.register_blueprint(match_bp)

    # Initialize Database and FTS table
    with app.app_context():
        db.create_all()
        from services import vector_service
        vector_service.init_fts_table()

    # Context Processors
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

    # Template Filters
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

# Expose the app object for Gunicorn
app = create_app()

if __name__ == "__main__":
    # This block is ignored by Gunicorn in production, but used for local testing
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
