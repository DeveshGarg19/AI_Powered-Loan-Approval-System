# 🏦 LoanIQ - AI-Powered Loan Approval System

**LoanIQ** is an intelligent web application built with Flask that utilizes Machine Learning to predict whether a loan application will be approved or rejected. Not only does it provide a prediction, but it also uses **SHAP** (SHapley Additive exPlanations) to explain *why* a decision was made and offers actionable recommendations to the user.

---

## ✨ Features

- **🔐 User Authentication:** Secure registration and login with hashed passwords.
- **✉️ Email Verification (OTP):** Secure email verification during registration and password reset using auto-generated OTPs.
- **🤖 AI Loan Prediction:** Uses a trained Machine Learning model to evaluate applicants based on income, credit score, DTI ratio, and more.
- **📊 Explainable AI (XAI):** Visualizes the top features impacting the loan decision using SHAP values and dynamically generates a personalized bar chart.
- **💡 Actionable Recommendations:** Gives users tailored advice on how to improve their chances of approval (e.g., "Reduce your DTI Ratio").
- **📧 Automated Notifications:** Sends real-time email notifications regarding application results.
- **📁 User Dashboard:** Allows users to view their past loan application history, approval rates, and average probabilities.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, Flask-SQLAlchemy, Flask-Mail
- **Frontend:** HTML, CSS, JavaScript (Jinja2 Templates)
- **Database:** SQLite
- **Machine Learning:** Scikit-Learn, Pandas, NumPy
- **Explainability & Visuals:** SHAP, Matplotlib (base64 encoded images)

---

## 🚀 Installation & Setup

Follow these steps to run the project locally on your machine.

### 1. Clone the repository
```bash
git clone https://github.com/DeveshGarg19/AI_Powered-Loan-Approval-System.git
cd AI_Powered-Loan-Approval-System
```

### 2. Create a Virtual Environment (Optional but recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
Make sure you have all required Python libraries installed:
```bash
pip install Flask Flask-SQLAlchemy Flask-Mail Werkzeug pandas numpy scikit-learn shap matplotlib
```
*(Alternatively, if you have a `requirements.txt`, run `pip install -r requirements.txt`)*

### 4. Configure Email (Gmail App Password)
For the OTP and email notifications to work, you need to update the email credentials in `app.py` (around line 38):
1. Go to your Google Account -> Security -> 2-Step Verification.
2. Scroll down to **App passwords** and generate a new password for this app.
3. Replace the `MAIL_USERNAME` and `MAIL_PASSWORD` in `app.py` with your email and the 16-character app password.

### 5. Run the Application
Start the Flask development server:
```bash
python app.py
```
The application will automatically open in your default web browser at `http://127.0.0.1:5000`.

---

## 📂 Project Structure

```text
📁 AI_Powered-Loan-Approval-System
├── 📁 ML_Model/            # Contains the trained .pkl files (model, scaler, encoder)
├── 📁 static/              # CSS styles (Styles.css) and static assets
├── 📁 templates/           # HTML templates (index, login, dashboard, result, etc.)
├── 📄 app.py               # Main Flask application and routing logic
├── 📄 database.db          # SQLite Database (generated automatically)
├── 📄 loan_approval_data.csv # Dataset used for the ML Model
├── 📄 proj.ipynb           # Jupyter Notebook with ML training and EDA
└── 📄 README.md            # Project documentation
```

---

## 🧠 How the ML Model Works
1. **Inputs:** The user submits a form with their details (Income, Loan Amount, Credit Score, etc.).
2. **Preprocessing:** The app scales numerical data and applies One-Hot Encoding to categorical data using the saved `scaler.pkl` and `ohe_encoder.pkl`.
3. **Prediction:** The model calculates a probability. A threshold of `0.45` is used (≥45% = Approved).
4. **Explanation:** `shap.LinearExplainer` breaks down the prediction, identifying exactly which features pulled the score down or pushed it up.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📝 License
This project is open-source and available under the [MIT License](LICENSE).
