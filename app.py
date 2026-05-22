from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required
import joblib
import os
import json
import numpy as np

from config import Config
from database import db, User


# -------- GLOBALS --------
loaded_ml_models = {}
loaded_label_encoder = None
loaded_vectorizer = None
loaded_patterns_data = {}


# -------- RULE BASED FUNCTION --------
def detect_pattern_rule_based(code):
    code = code.lower()

    if "context" in code and "strategy" in code:
        return "Strategy", 90

    if "notify" in code or "update" in code:
        return "Observer", 85

    if "create" in code:
        return "Factory Method", 85

    if "instance" in code:
        return "Singleton", 90

    if "wrap" in code:
        return "Decorator", 85

    if "adapter" in code:
        return "Adapter", 85

    if "execute" in code:
        return "Command", 85

    return None, 0


# -------- FEATURE EXTRACTION --------
def extract_features(code):
    code = code.lower()

    return [
        code.count("class"),
        code.count("def"),
        code.count("self"),
        code.count("interface"),
        code.count("extends"),
        code.count("implements"),

        1 if "context" in code else 0,
        1 if "strategy" in code else 0,
        1 if "execute" in code else 0,

        1 if "observer" in code else 0,
        1 if "notify" in code else 0,
        1 if "update" in code else 0,

        1 if "create" in code else 0,
        1 if "factory" in code else 0,

        1 if "instance" in code else 0,
        1 if "private constructor" in code else 0,

        1 if "wrap" in code else 0,
        1 if "adapter" in code else 0,
        1 if "command" in code else 0
    ]


# -------- LOAD MODELS --------
def load_ml(app):
    global loaded_ml_models, loaded_label_encoder
    global loaded_vectorizer, loaded_patterns_data

    loaded_label_encoder = joblib.load(app.config['LABEL_ENCODER_PATH'])
    loaded_vectorizer = joblib.load(app.config['TFIDF_VECTORIZER_PATH'])

    for name, file in app.config['ML_MODELS'].items():
        path = os.path.join(app.config['MODEL_DIR'], file)
        if os.path.exists(path):
            loaded_ml_models[name] = joblib.load(path)

    with open(app.config['DESIGN_PATTERNS_DATA'], 'r', encoding='utf-8') as f:
        loaded_patterns_data = json.load(f)


# -------- APP --------
def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    db.init_app(app)

    login_manager = LoginManager(app)
    login_manager.login_view = 'login'

    with app.app_context():
        db.create_all()

    load_ml(app)

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))


    # -------- HOME --------
    @app.route('/')
    def home():
        return render_template('index.html')


    # -------- REGISTER --------
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username = request.form.get('username')
            password = request.form.get('password')

            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                flash("User already exists", "danger")
                return redirect(url_for('register'))

            new_user = User(username=username)
            new_user.set_password(password)

            db.session.add(new_user)
            db.session.commit()

            flash("Registration successful!", "success")
            return redirect(url_for('login'))

        return render_template('register.html')


    # -------- LOGIN --------
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            user = User.query.filter_by(username=request.form.get('username')).first()
            if user and user.check_password(request.form.get('password')):
                login_user(user)
                return redirect(url_for('predict'))
            flash("Invalid login", "danger")
        return render_template('login.html')


    # -------- LOGOUT --------
    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        return redirect(url_for('home'))


    # -------- PREDICT --------
    @app.route('/predict', methods=['GET', 'POST'])
    @login_required
    def predict():
        prediction_results = None
        models = list(loaded_ml_models.keys())

        if request.method == 'POST':
            try:
                text = request.form.get('characteristics', '')
                code = request.form.get('code', '')
                file = request.files.get('file')

                file_text = ""
                if file and file.filename:
                    try:
                        file_text = file.read().decode('utf-8')
                    except:
                        file_text = ""

                # Combine input
                final_input = " ".join([text, code, file_text]).lower().strip()

                # RULE-BASED FIRST
                rule_pattern, rule_conf = detect_pattern_rule_based(final_input)

                # Model selection
                model_name = request.form.get('model_selector') or models[0]
                model = loaded_ml_models[model_name]

                # TF-IDF
                tfidf = loaded_vectorizer.transform([final_input]).toarray()

                # Structural features
                struct_features = np.array(extract_features(final_input)).reshape(1, -1)

                # Combine
                features = np.hstack((tfidf, struct_features))

                # Normalize
                features = features / (np.linalg.norm(features) + 1e-6)

                # FINAL PREDICTION
                if rule_pattern:
                    pattern = rule_pattern
                    confidence = f"{rule_conf}%"
                else:
                    pred = model.predict(features)[0]
                    pattern = loaded_label_encoder.inverse_transform([pred])[0]

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(features)[0]
                        confidence = f"{np.max(probs)*100:.2f}%"
                    else:
                        confidence = "85%"

                # Explanation
                info = loaded_patterns_data.get(pattern, {})
                explanation = info.get("explanation_text", "No explanation available")

                diagram = url_for('static', filename=f'uml_diagrams/{pattern}.png')

                prediction_results = {
                    "pattern_name": pattern,
                    "confidence": confidence,
                    "explanation": explanation,
                    "diagram_path": diagram,
                    "model_used": model_name.upper()
                }

            except Exception as e:
                flash(f"Error: {e}", "danger")

        return render_template(
            "predict.html",
            prediction_results=prediction_results,
            available_models=models
        )

    return app


# -------- RUN --------
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)