import sys, subprocess, os, secrets, hmac, math

from flask import Flask, render_template, request, redirect, url_for, session, jsonify, abort
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from markupsafe import escape
from werkzeug.security import generate_password_hash, check_password_hash
import string
from datetime import datetime, timedelta
from functools import wraps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_dotenv(env_path):
    """Load simple KEY=VALUE pairs from a local .env file without overriding existing env vars."""
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, val = line.split('=', 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, val)

_load_dotenv(os.path.join(BASE_DIR, '.env'))

app = Flask(__name__)

# ─── Config ─────────────────────────────────────────────────────────────────
_secret_key = os.environ.get("SECRET_KEY")
if not _secret_key:
    print("⚠️ WARNING: SECRET_KEY environment variable is not set; using an ephemeral key.")
    _secret_key = secrets.token_hex(32)
app.secret_key = _secret_key

app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = bool(
    os.environ.get("VERCEL") or os.environ.get("SESSION_COOKIE_SECURE", "").lower() == "true"
)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=2)

# Database Configuration
# Uses PostgreSQL if DATABASE_URL is present (perfect for Vercel persistence),
# otherwise falls back to local SQLite for offline development.
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    DATABASE_URL = DATABASE_URL.strip().strip('"').strip("'")
    # Explicitly select the psycopg2 driver (psycopg2-binary in requirements.txt)
    for prefix in ("postgresql+psycopg://", "postgres+psycopg://", "postgresql://", "postgres://"):
        if DATABASE_URL.startswith(prefix):
            DATABASE_URL = "postgresql+psycopg2://" + DATABASE_URL[len(prefix):]
            break
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
else:
    if os.environ.get("VERCEL"):
        db_path = '/tmp/database.db'
        src_db = os.path.join(BASE_DIR, 'database.db')
        if not os.path.exists(db_path) and os.path.exists(src_db):
            import shutil
            shutil.copy(src_db, db_path)
    else:
        db_path = os.path.join(BASE_DIR, 'database.db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# ─── Mail Config (loaded from environment) ──────────────────────────────────
app.config['MAIL_SERVER']   = os.environ.get('MAIL_SERVER') or 'smtp.gmail.com'
app.config['MAIL_PORT']     = int(os.environ.get('MAIL_PORT') or 587)
app.config['MAIL_USE_TLS']  = (os.environ.get('MAIL_USE_TLS') or 'true').lower() in ('1', 'true', 'yes')
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', '')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', '')

db   = SQLAlchemy(app)
mail = Mail(app)

# ─── Database Models ─────────────────────────────────────────────────────────
class User(db.Model):
    id           = db.Column(db.Integer, primary_key=True)
    name         = db.Column(db.String(100), nullable=False)
    email        = db.Column(db.String(120), unique=True, nullable=False)
    password     = db.Column(db.String(200), nullable=False)
    is_verified  = db.Column(db.Boolean, default=False)
    otp          = db.Column(db.String(6), nullable=True)
    otp_expiry   = db.Column(db.DateTime, nullable=True)
    created_at   = db.Column(db.DateTime, default=datetime.utcnow)
    applications = db.relationship('Application', backref='user', lazy=True)

class Application(db.Model):
    id          = db.Column(db.Integer, primary_key=True)
    user_id     = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    approved    = db.Column(db.Boolean, nullable=False)
    probability = db.Column(db.Float, nullable=False)
    income      = db.Column(db.Float)
    loan_amount = db.Column(db.Float)
    credit_score= db.Column(db.Float)
    dti_ratio   = db.Column(db.Float)
    loan_purpose= db.Column(db.String(50))
    reasons     = db.Column(db.Text)
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables
with app.app_context():
    db.create_all()

# ─── Load ML Models (Lazy Loading) ───────────────────────────────────────────
# Models are loaded lazily and cached globally to prevent Vercel Serverless Function timeouts
_ml_model = None
_ml_scaler = None
_ml_ohe = None
_ml_feature_names = None

def get_ml_resources():
    global _ml_model, _ml_scaler, _ml_ohe, _ml_feature_names
    if _ml_model is None:
        import pickle
        SAVE_PATH = os.path.join(BASE_DIR, 'Ml_Model')
        _ml_model = pickle.load(open(os.path.join(SAVE_PATH, 'model.pkl'), 'rb'))
        _ml_scaler = pickle.load(open(os.path.join(SAVE_PATH, 'scaler.pkl'), 'rb'))
        _ml_ohe = pickle.load(open(os.path.join(SAVE_PATH, 'ohe_encoder.pkl'), 'rb'))
        _ml_feature_names = pickle.load(open(os.path.join(SAVE_PATH, 'feature_names.pkl'), 'rb'))
        print("✅ Lazy-loaded all machine learning models successfully!")
    return _ml_model, _ml_scaler, _ml_ohe, _ml_feature_names

# ─── Jinja2 Filters ──────────────────────────────────────────────────────────
IST_OFFSET = timedelta(hours=5, minutes=30)

@app.template_filter('to_ist')
def to_ist(dt):
    """Convert a naive UTC datetime to IST (UTC+5:30)."""
    if dt is None:
        return ''
    return dt + IST_OFFSET

# ─── Security & Helper Functions ─────────────────────────────────────────────
MAX_OTP_ATTEMPTS = 5
MIN_PASSWORD_LEN = 8

def get_csrf_token():
    token = session.get('_csrf_token')
    if not token:
        token = secrets.token_urlsafe(32)
        session['_csrf_token'] = token
    return token

@app.context_processor
def inject_csrf_token():
    return dict(csrf_token=get_csrf_token)

@app.before_request
def verify_csrf():
    if request.method == 'POST':
        session_token = session.get('_csrf_token')
        form_token = request.form.get('csrf_token') or request.headers.get('X-CSRFToken')
        if not session_token or not form_token or not hmac.compare_digest(session_token, form_token):
            abort(403)

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def generate_otp():
    return ''.join(secrets.choice(string.digits) for _ in range(6))

def _verify_user_otp(user, entered_otp, attempts_key):
    """Validate OTP with expiry check, constant-time comparison, and attempt limit."""
    if not user or not user.otp or not user.otp_expiry:
        return 'invalid'
    if datetime.utcnow() > user.otp_expiry:
        user.otp = None
        user.otp_expiry = None
        db.session.commit()
        session.pop(attempts_key, None)
        return 'expired'
    if not hmac.compare_digest(user.otp, entered_otp.strip()):
        attempts = session.get(attempts_key, 0) + 1
        session[attempts_key] = attempts
        if attempts >= MAX_OTP_ATTEMPTS:
            user.otp = None
            user.otp_expiry = None
            db.session.commit()
            session.pop(attempts_key, None)
            return 'locked'
        return 'mismatch'
    session.pop(attempts_key, None)
    return 'ok'

def _ensure_mail_configured():
    if not app.config.get('MAIL_USERNAME') or not app.config.get('MAIL_PASSWORD'):
        raise RuntimeError("MAIL_USERNAME and MAIL_PASSWORD must be configured in environment variables.")

def send_otp_email(email, name, otp, subject="Verify Your Email"):
    _ensure_mail_configured()
    safe_name = escape(name)
    safe_subject = escape(subject)
    msg = Message(subject,
        sender=app.config['MAIL_USERNAME'],
        recipients=[email])
    msg.html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 500px; margin: 0 auto; padding: 30px; background: #f9f9ff; border-radius: 16px;">
        <h2 style="color: #1a1a2e; text-align: center;">🏦 LoanIQ</h2>
        <p style="color: #555;">Hello <b>{safe_name}</b>,</p>
        <p style="color: #555;">{safe_subject}. Use the OTP below:</p>
        <div style="text-align: center; margin: 30px 0;">
            <span style="font-size: 36px; font-weight: bold; letter-spacing: 10px;
                color: #6c63ff; background: #efefff; padding: 16px 32px;
                border-radius: 12px; display: inline-block;">{escape(otp)}</span>
        </div>
        <p style="color: #888; font-size: 13px; text-align: center;">
            This OTP is valid for <b>10 minutes</b>. Do not share it with anyone.
        </p>
        <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;"/>
        <p style="color: #aaa; font-size: 12px; text-align: center;">LoanIQ — AI Loan Approval System</p>
    </div>
    """
    mail.send(msg)

def send_approval_email(email, name, approved, probability):
    _ensure_mail_configured()
    safe_name = escape(name)
    status = "✅ Approved" if approved else "❌ Rejected"
    color  = "#00d68f" if approved else "#ff4d6d"
    msg    = Message(f"LoanIQ — Your Loan Application Result: {status}",
        sender=app.config['MAIL_USERNAME'],
        recipients=[email])
    msg.html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 500px; margin: 0 auto; padding: 30px; background: #f9f9ff; border-radius: 16px;">
        <h2 style="color: #1a1a2e; text-align: center;">🏦 LoanIQ</h2>
        <p style="color: #555;">Hello <b>{safe_name}</b>,</p>
        <p style="color: #555;">Your loan application has been analyzed. Here is the result:</p>
        <div style="text-align: center; margin: 24px 0; padding: 20px;
            background: {'#e6fff5' if approved else '#fff0f3'};
            border: 2px solid {color}; border-radius: 12px;">
            <div style="font-size: 28px; font-weight: bold; color: {color};">{status}</div>
            <div style="font-size: 16px; color: #555; margin-top: 8px;">
                Approval Probability: <b>{probability}%</b>
            </div>
        </div>
        <p style="color: #555;">Log in to your LoanIQ dashboard to view full details.</p>
        <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;"/>
        <p style="color: #aaa; font-size: 12px; text-align: center;">LoanIQ — AI Loan Approval System</p>
    </div>
    """
    mail.send(msg)

# ─── Auth Routes ─────────────────────────────────────────────────────────────

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name     = request.form.get('name', '').strip()
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not name or not email:
            return render_template('register.html', error="Name and email are required.")
        if len(password) < MIN_PASSWORD_LEN:
            return render_template('register.html', error=f"Password must be at least {MIN_PASSWORD_LEN} characters long.")
        if User.query.filter_by(email=email).first():
            return render_template('register.html', error="Email already registered!")

        otp        = generate_otp()
        otp_expiry = datetime.utcnow() + timedelta(minutes=10)
        hashed_pw  = generate_password_hash(password)

        user = User(name=name, email=email, password=hashed_pw,
                    otp=otp, otp_expiry=otp_expiry)
        db.session.add(user)
        try:
            send_otp_email(email, name, otp, "Verify Your LoanIQ Account")
            db.session.commit()
        except Exception as exc:
            db.session.rollback()
            print(f"Registration email error: {exc}")
            return render_template('register.html', error="Unable to send verification email right now. Please try again later.")

        session['verify_email'] = email
        session.pop('verify_otp_attempts', None)
        return redirect(url_for('verify_otp'))

    return render_template('register.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    email = session.get('verify_email')
    if not email:
        return redirect(url_for('register'))

    if request.method == 'POST':
        entered_otp = request.form.get('otp', '')
        user        = User.query.filter_by(email=email).first()
        status      = _verify_user_otp(user, entered_otp, 'verify_otp_attempts')

        if status == 'invalid':
            return render_template('verify_otp.html', email=email, error="No active OTP found. Please request a new verification code.")
        if status == 'expired':
            return render_template('verify_otp.html', email=email, error="OTP expired. Please request a new verification code.")
        if status == 'locked':
            return render_template('verify_otp.html', email=email, error="Too many incorrect attempts. OTP invalidated.")
        if status == 'mismatch':
            return render_template('verify_otp.html', email=email, error="Incorrect OTP. Try again.")

        user.is_verified = True
        user.otp         = None
        user.otp_expiry  = None
        db.session.commit()
        session.pop('verify_email', None)
        return redirect(url_for('login', success="Email verified! Please login."))

    return render_template('verify_otp.html', email=email)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        user     = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password, password):
            return render_template('login.html', error="Invalid email or password.")
        if not user.is_verified:
            session['verify_email'] = email
            return redirect(url_for('verify_otp'))

        session.clear()
        session.permanent = True
        session['_csrf_token'] = secrets.token_urlsafe(32)
        session['user_id']     = user.id
        session['user_name']   = user.name
        session['user_email']  = user.email
        return redirect(url_for('apply'))

    return render_template('login.html', success=request.args.get('success'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        user  = User.query.filter_by(email=email).first()
        if user:
            otp              = generate_otp()
            user.otp         = otp
            user.otp_expiry  = datetime.utcnow() + timedelta(minutes=10)
            db.session.commit()
            try:
                send_otp_email(email, user.name, otp, "Reset Your LoanIQ Password")
            except Exception as exc:
                print(f"Password reset email error: {exc}")

        session['reset_email'] = email
        session.pop('reset_otp_attempts', None)
        return redirect(url_for('reset_password'))

    return render_template('forgot_password.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    email = session.get('reset_email')
    if not email:
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        otp          = request.form.get('otp', '')
        new_password = request.form.get('new_password', '')
        if len(new_password) < MIN_PASSWORD_LEN:
            return render_template('reset_password.html', error=f"Password must be at least {MIN_PASSWORD_LEN} characters long.")

        user   = User.query.filter_by(email=email).first()
        status = _verify_user_otp(user, otp, 'reset_otp_attempts')

        if status in ('invalid', 'mismatch'):
            return render_template('reset_password.html', error="Incorrect or invalid OTP.")
        if status == 'expired':
            return render_template('reset_password.html', error="OTP expired. Please request a new reset code.")
        if status == 'locked':
            return render_template('reset_password.html', error="Too many incorrect attempts. Please request a new reset code.")

        user.password   = generate_password_hash(new_password)
        user.otp        = None
        user.otp_expiry = None
        db.session.commit()
        session.pop('reset_email', None)
        return redirect(url_for('login', success="Password reset successful!"))

    return render_template('reset_password.html')

# ─── Main Routes ─────────────────────────────────────────────────────────────

@app.route('/')
def home():
    logged_in = 'user_id' in session
    return render_template('landing.html', logged_in=logged_in, user_name=session.get('user_name'))

@app.route('/apply')
@login_required
def apply():
    return render_template('index.html', user_name=session.get('user_name'))

@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session['user_id']
    apps    = Application.query.filter_by(user_id=user_id).order_by(Application.created_at.desc()).all()
    total   = len(apps)
    approved= sum(1 for a in apps if a.approved)
    rejected= total - approved
    avg_prob= round(sum(a.probability for a in apps) / total, 1) if total > 0 else 0
    return render_template('dashboard.html',
        user_name=session.get('user_name'),
        user_email=session.get('user_email'),
        applications=apps,
        total=total, approved=approved,
        rejected=rejected, avg_prob=avg_prob
    )

@app.route('/emi')
@login_required
def emi():
    return render_template('emi.html', user_name=session.get('user_name'))

def _parse_finite_float(data, field, min_val=None, max_val=None):
    raw_val = data.get(field)
    if raw_val is None or str(raw_val).strip() == '':
        raise ValueError(f"{field.replace('_', ' ')} is required.")
    val = float(raw_val)
    if not math.isfinite(val):
        raise ValueError(f"{field.replace('_', ' ')} must be a valid finite number.")
    if min_val is not None and val < min_val:
        raise ValueError(f"{field.replace('_', ' ')} must be at least {min_val}.")
    if max_val is not None and val > max_val:
        raise ValueError(f"{field.replace('_', ' ')} must be at most {max_val}.")
    return val

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Lazy imports to optimize serverless cold-start performance
        import pandas as pd
        import numpy as np
        import shap

        # Load ML resources lazily
        model, scaler, ohe, feature_names = get_ml_resources()

        data = request.form

        age              = _parse_finite_float(data, 'Age', 18, 100)
        dependents       = _parse_finite_float(data, 'Dependents', 0, 20)
        applicant_income = _parse_finite_float(data, 'Applicant_Income', 0)
        loan_amount      = _parse_finite_float(data, 'Loan_Amount', 0)
        loan_term        = _parse_finite_float(data, 'Loan_Term', 1, 480)
        credit_score     = _parse_finite_float(data, 'Credit_Score', 300, 900)
        dti_ratio        = _parse_finite_float(data, 'DTI_Ratio', 0.0, 1.0)
        savings          = _parse_finite_float(data, 'Savings', 0)

        education_level = data.get('Education_Level', '')
        if education_level not in ('Graduate', 'Not Graduate'):
            raise ValueError("Invalid Education Level selected.")

        raw = pd.DataFrame([{
            'Gender':            data.get('Gender', ''),
            'Age':               age,
            'Marital_Status':    data.get('Marital_Status', ''),
            'Dependents':        dependents,
            'Education_Level':   education_level,
            'Employment_Status': data.get('Employment_Status', ''),
            'Employer_Category': data.get('Employer_Category', ''),
            'Applicant_Income':  applicant_income,
            'Loan_Amount':       loan_amount,
            'Loan_Term':         loan_term,
            'Loan_Purpose':      data.get('Loan_Purpose', ''),
            'Property_Area':     data.get('Property_Area', ''),
            'Credit_Score':      credit_score,
            'DTI_Ratio':         dti_ratio,
            'Savings':           savings,
        }])

        education_map = {'Graduate': 1, 'Not Graduate': 0}
        raw['Education_Level'] = raw['Education_Level'].map(education_map)

        cat_cols   = ['Employment_Status','Marital_Status','Loan_Purpose','Gender','Property_Area','Employer_Category']
        encoded    = ohe.transform(raw[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_cols))
        raw        = pd.concat([raw.drop(columns=cat_cols), encoded_df], axis=1)

        raw['DTI_Ratio_sq']    = raw['DTI_Ratio'] ** 2
        raw['Credit_Score_sq'] = raw['Credit_Score'] ** 2
        raw = raw.drop(columns=['DTI_Ratio','Credit_Score'], errors='ignore')
        raw = raw.reindex(columns=feature_names, fill_value=0)

        scaled      = scaler.transform(raw)
        THRESHOLD   = 0.45
        probability = float(round(model.predict_proba(scaled)[0][1] * 100, 1))
        prediction  = 1 if (probability / 100) >= THRESHOLD else 0

        background = np.zeros((1, len(feature_names)))
        masker     = shap.maskers.Independent(background)
        explainer  = shap.LinearExplainer(model, masker=masker)
        shap_vals  = explainer.shap_values(scaled)
        shap_dict  = dict(zip(feature_names, shap_vals[0].tolist()))

        reasons = []
        for feat, val in sorted(shap_dict.items(), key=lambda x: x[1]):
            if val < 0:
                reasons.append(f"{feat.replace('_',' ').title()} is negatively impacting your approval")

        recommendations = []
        if shap_dict.get('DTI_Ratio_sq', 0) < 0:
            recommendations.append("📉 Reduce your DTI Ratio — pay off existing debts.")
        if shap_dict.get('Credit_Score_sq', 0) < 0:
            recommendations.append("📊 Improve your Credit Score — pay bills on time.")
        if shap_dict.get('Loan_Amount', 0) < 0:
            recommendations.append("💰 Reduce Loan Amount — apply for a smaller amount.")
        if shap_dict.get('Loan_Term', 0) < 0:
            recommendations.append("📅 Reduce Loan Term — choose a shorter repayment period.")

        # Save to database
        app_record = Application(
            user_id     = session['user_id'],
            approved    = bool(prediction),
            probability = probability,
            income      = applicant_income,
            loan_amount = loan_amount,
            credit_score= credit_score,
            dti_ratio   = dti_ratio,
            loan_purpose= data.get('Loan_Purpose', ''),
            reasons     = ', '.join(reasons[:3])
        )
        db.session.add(app_record)
        db.session.commit()

        # Send result email
        try:
            send_approval_email(session['user_email'], session['user_name'], bool(prediction), probability)
        except Exception as mail_exc:
            print(f"Approval email skipped/failed: {mail_exc}")

        # Extract features and SHAP values for modern dynamic browser-side rendering (Chart.js/ApexCharts)
        # This completely replaces the heavy matplotlib server-side rendering, saving 50+ MB of bundle space!
        sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        sorted_shap = sorted(sorted_shap, key=lambda x: x[1])  # Sort ascending for horizontal bar chart
        
        chart_features = [x[0].replace('_', ' ').title() for x in sorted_shap]
        chart_values = [round(x[1], 4) for x in sorted_shap]

        return render_template('result.html',
            approved=bool(prediction), probability=probability,
            reasons=reasons, recommendations=recommendations,
            chart_features=chart_features, chart_values=chart_values,
            user_name=session.get('user_name')
        )

    except ValueError as ve:
        return render_template('index.html', error=str(ve), user_name=session.get('user_name'))
    except Exception:
        import traceback
        print(traceback.format_exc())
        return render_template(
            'index.html',
            error="Unable to analyze your application right now. Please verify your inputs and try again.",
            user_name=session.get('user_name')
        )

if __name__ == '__main__':
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    if os.environ.get("FLASK_RUNNING") != "1":
        os.environ["FLASK_RUNNING"] = "1"
        import webbrowser
        from threading import Timer
        Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
        cmd = [sys.executable, "-m", "flask", "--app", __file__, "run"]
        if debug_mode:
            cmd.append("--debug")
        sys.exit(subprocess.call(cmd, shell=False))
    app.run(debug=debug_mode)