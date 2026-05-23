import sys, subprocess, os

from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
import base64, io, random, string
from datetime import datetime, timedelta, timezone
from functools import wraps

app = Flask(__name__)

# ─── Config ─────────────────────────────────────────────────────────────────
app.secret_key = "loaniq_secret_key_2024"

# Database Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Uses PostgreSQL if DATABASE_URL is present (perfect for Vercel persistence),
# otherwise falls back to local SQLite for offline development.
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL:
    # Fix potential postgres:// to postgresql:// scheme issue for SQLAlchemy
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
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

# ─── Gmail Config ────────────────────────────────────────────────────────────
# IMPORTANT: Replace with your Gmail and App Password
app.config['MAIL_SERVER']   = 'smtp.gmail.com'
app.config['MAIL_PORT']     = 587
app.config['MAIL_USE_TLS']  = True
app.config['MAIL_USERNAME'] = 'garg.devesh0619@gmail.com'       # ← Your Gmail
app.config['MAIL_PASSWORD'] = 'qtce qrrh ilcv lmvg'     # ← Gmail App Password

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

# ─── Helper Functions ────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

def send_otp_email(email, name, otp, subject="Verify Your Email"):
    msg = Message(subject,
        sender=app.config['MAIL_USERNAME'],
        recipients=[email])
    msg.html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 500px; margin: 0 auto; padding: 30px; background: #f9f9ff; border-radius: 16px;">
        <h2 style="color: #1a1a2e; text-align: center;">🏦 LoanIQ</h2>
        <p style="color: #555;">Hello <b>{name}</b>,</p>
        <p style="color: #555;">{subject}. Use the OTP below:</p>
        <div style="text-align: center; margin: 30px 0;">
            <span style="font-size: 36px; font-weight: bold; letter-spacing: 10px;
                color: #6c63ff; background: #efefff; padding: 16px 32px;
                border-radius: 12px; display: inline-block;">{otp}</span>
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
    status = "✅ Approved" if approved else "❌ Rejected"
    color  = "#00d68f" if approved else "#ff4d6d"
    msg    = Message(f"LoanIQ — Your Loan Application Result: {status}",
        sender=app.config['MAIL_USERNAME'],
        recipients=[email])
    msg.html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 500px; margin: 0 auto; padding: 30px; background: #f9f9ff; border-radius: 16px;">
        <h2 style="color: #1a1a2e; text-align: center;">🏦 LoanIQ</h2>
        <p style="color: #555;">Hello <b>{name}</b>,</p>
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
        name     = request.form['name']
        email    = request.form['email']
        password = request.form['password']

        if User.query.filter_by(email=email).first():
            return render_template('register.html', error="Email already registered!")

        otp        = generate_otp()
        otp_expiry = datetime.utcnow() + timedelta(minutes=10)
        hashed_pw  = generate_password_hash(password)

        user = User(name=name, email=email, password=hashed_pw,
                    otp=otp, otp_expiry=otp_expiry)
        db.session.add(user)
        db.session.commit()

        send_otp_email(email, name, otp, "Verify Your LoanIQ Account")
        session['verify_email'] = email
        return redirect(url_for('verify_otp'))

    return render_template('register.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    email = session.get('verify_email')
    if not email:
        return redirect(url_for('register'))

    if request.method == 'POST':
        entered_otp = request.form['otp']
        user        = User.query.filter_by(email=email).first()

        if not user:
            return render_template('verify_otp.html', error="User not found.")
        if datetime.utcnow() > user.otp_expiry:
            return render_template('verify_otp.html', error="OTP expired. Please register again.")
        if user.otp != entered_otp:
            return render_template('verify_otp.html', error="Incorrect OTP. Try again.")

        user.is_verified = True
        user.otp         = None
        db.session.commit()
        session.pop('verify_email', None)
        return redirect(url_for('login', success="Email verified! Please login."))

    return render_template('verify_otp.html', email=email)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form['email']
        password = request.form['password']
        user     = User.query.filter_by(email=email).first()

        if not user or not check_password_hash(user.password, password):
            return render_template('login.html', error="Invalid email or password.")
        if not user.is_verified:
            session['verify_email'] = email
            return redirect(url_for('verify_otp'))

        session['user_id']   = user.id
        session['user_name'] = user.name
        session['user_email']= user.email
        return redirect(url_for('home'))

    return render_template('login.html', success=request.args.get('success'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user  = User.query.filter_by(email=email).first()
        if not user:
            return render_template('forgot_password.html', error="Email not found.")

        otp              = generate_otp()
        user.otp         = otp
        user.otp_expiry  = datetime.utcnow() + timedelta(minutes=10)
        db.session.commit()
        send_otp_email(email, user.name, otp, "Reset Your LoanIQ Password")
        session['reset_email'] = email
        return redirect(url_for('reset_password'))

    return render_template('forgot_password.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    email = session.get('reset_email')
    if not email:
        return redirect(url_for('forgot_password'))

    if request.method == 'POST':
        otp          = request.form['otp']
        new_password = request.form['new_password']
        user         = User.query.filter_by(email=email).first()

        if not user or user.otp != otp:
            return render_template('reset_password.html', error="Incorrect OTP.")
        if datetime.utcnow() > user.otp_expiry:
            return render_template('reset_password.html', error="OTP expired.")

        user.password = generate_password_hash(new_password)
        user.otp      = None
        db.session.commit()
        session.pop('reset_email', None)
        return redirect(url_for('login', success="Password reset successful!"))

    return render_template('reset_password.html')

# ─── Main Routes ─────────────────────────────────────────────────────────────

@app.route('/')
@login_required
def home():
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

        raw = pd.DataFrame([{
            'Gender':            data['Gender'],
            'Age':               float(data['Age']),
            'Marital_Status':    data['Marital_Status'],
            'Dependents':        float(data['Dependents']),
            'Education_Level':   data['Education_Level'],
            'Employment_Status': data['Employment_Status'],
            'Employer_Category': data['Employer_Category'],
            'Applicant_Income':  float(data['Applicant_Income']),
            'Loan_Amount':       float(data['Loan_Amount']),
            'Loan_Term':         float(data['Loan_Term']),
            'Loan_Purpose':      data['Loan_Purpose'],
            'Property_Area':     data['Property_Area'],
            'Credit_Score':      float(data['Credit_Score']),
            'DTI_Ratio':         float(data['DTI_Ratio']),
            'Savings':           float(data['Savings']),
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
            income      = float(data['Applicant_Income']),
            loan_amount = float(data['Loan_Amount']),
            credit_score= float(data['Credit_Score']),
            dti_ratio   = float(data['DTI_Ratio']),
            loan_purpose= data['Loan_Purpose'],
            reasons     = ', '.join(reasons[:3])
        )
        db.session.add(app_record)
        db.session.commit()

        # Send result email
        try:
            send_approval_email(session['user_email'], session['user_name'], bool(prediction), probability)
        except:
            pass  # Don't fail if email fails

        # Extract features and SHAP values for modern dynamic browser-side rendering (Chart.js/ApexCharts)
        # This completely replaces the heavy matplotlib server-side rendering, saving 50+ MB of bundle space!
        sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        sorted_shap = sorted(sorted_shap, key=lambda x: x[1])  # Sort ascending for horizontal bar chart
        
        chart_features = [x[0].replace('_', ' ').title() for x in sorted_shap]
        chart_values = [round(x[1], 4) for x in sorted_shap]

        session['result'] = {
            'approved': bool(prediction), 'probability': probability,
            'reasons': reasons, 'recommendations': recommendations,
            'shap_dict': shap_dict, 'form_data': dict(data)
        }

        return render_template('result.html',
            approved=bool(prediction), probability=probability,
            reasons=reasons, recommendations=recommendations,
            chart_features=chart_features, chart_values=chart_values,
            user_name=session.get('user_name')
        )

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return render_template('index.html', error=str(e), user_name=session.get('user_name'))

if __name__ == '__main__':
    if os.environ.get("FLASK_RUNNING") != "1":
        os.environ["FLASK_RUNNING"] = "1"
        import webbrowser
        from threading import Timer
        Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
        sys.exit(subprocess.call(
            ["C:/Program Files/Python313/python.exe", "-m", "flask", "--app", __file__, "run", "--debug"],
            shell=False
        ))
    app.run(debug=True)