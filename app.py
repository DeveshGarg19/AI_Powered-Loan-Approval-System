import sys
import subprocess
import os

# Auto-launch Flask if run directly
if os.environ.get("FLASK_RUNNING") != "1":
    os.environ["FLASK_RUNNING"] = "1"
    script = os.path.abspath(__file__)
    import webbrowser
    from threading import Timer
    def open_browser():
        webbrowser.open("http://127.0.0.1:5000")
    Timer(1.5, open_browser).start()
    sys.exit(subprocess.call(
        ["C:/Program Files/Python313/python.exe", "-m", "flask", "--app", script, "run", "--debug"],
        shell=False
    ))

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Flask
import matplotlib.pyplot as plt
import base64
import io

app = Flask(__name__)

# ─── Load Models ────────────────────────────────────────────────────────────
SAVE_PATH = r'C:\Sem6_Proj\Ml_Model'

model         = pickle.load(open(os.path.join(SAVE_PATH, 'model.pkl'), 'rb'))
scaler        = pickle.load(open(os.path.join(SAVE_PATH, 'scaler.pkl'), 'rb'))
ohe           = pickle.load(open(os.path.join(SAVE_PATH, 'ohe_encoder.pkl'), 'rb'))
feature_names = pickle.load(open(os.path.join(SAVE_PATH, 'feature_names.pkl'), 'rb'))

print("✅ All models loaded successfully!")

# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.form

        # Build dataframe
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

        # Encode Education Level
        education_map = {'Graduate': 1, 'Not Graduate': 0}
        raw['Education_Level'] = raw['Education_Level'].map(education_map)

        # One Hot Encode
        cat_cols = ['Employment_Status', 'Marital_Status', 'Loan_Purpose',
                    'Gender', 'Property_Area', 'Employer_Category']
        encoded    = ohe.transform(raw[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(cat_cols))
        raw        = pd.concat([raw.drop(columns=cat_cols), encoded_df], axis=1)

        # Feature Engineering
        raw['DTI_Ratio_sq']    = raw['DTI_Ratio'] ** 2
        raw['Credit_Score_sq'] = raw['Credit_Score'] ** 2
        raw = raw.drop(columns=['DTI_Ratio', 'Credit_Score'], errors='ignore')

        # Align columns
        raw = raw.reindex(columns=feature_names, fill_value=0)

        # Scale
        scaled = scaler.transform(raw)

        # Predict
        prediction  = model.predict(scaled)[0]
        probability = round(model.predict_proba(scaled)[0][1] * 100, 1)

        # SHAP
        background = np.zeros((1, len(feature_names)))
        masker     = shap.maskers.Independent(background)
        explainer  = shap.LinearExplainer(model, masker=masker)
        shap_vals  = explainer.shap_values(scaled)
        shap_dict  = dict(zip(feature_names, shap_vals[0].tolist()))

        # Rejection Reasons
        reasons = []
        sorted_shap = sorted(shap_dict.items(), key=lambda x: x[1])
        for feat, val in sorted_shap:
            if val < 0:
                clean = feat.replace("_", " ").title()
                reasons.append(f"{clean} is negatively impacting your approval")

        # Recommendations
        recommendations = []
        if shap_dict.get('DTI_Ratio_sq', 0) < 0:
            recommendations.append("📉 Reduce your DTI Ratio — pay off existing debts before reapplying.")
        if shap_dict.get('Credit_Score_sq', 0) < 0:
            recommendations.append("📊 Improve your Credit Score — pay bills on time.")
        if shap_dict.get('Loan_Amount', 0) < 0:
            recommendations.append("💰 Reduce Loan Amount — apply for a smaller amount.")
        if shap_dict.get('Loan_Term', 0) < 0:
            recommendations.append("📅 Reduce Loan Term — choose a shorter repayment period.")

        # SHAP Chart
        shap_df = pd.DataFrame(list(shap_dict.items()), columns=["Feature", "SHAP Value"])
        shap_df["Feature"] = shap_df["Feature"].str.replace("_", " ")
        shap_df["abs"]     = shap_df["SHAP Value"].abs()
        shap_df            = shap_df.sort_values("abs", ascending=False).head(10)
        shap_df            = shap_df.sort_values("SHAP Value", ascending=True)

        colors = ["#ff4d6d" if v < 0 else "#00d68f" for v in shap_df["SHAP Value"]]

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        ax.barh(shap_df["Feature"], shap_df["SHAP Value"], color=colors)
        ax.axvline(0, color='white', linewidth=0.8, alpha=0.5)
        ax.tick_params(colors='white', labelsize=9)
        ax.spines[:].set_visible(False)
        ax.set_xlabel("SHAP Value", color='white', fontsize=10)
        ax.set_title("Top Feature Impacts", color='white', fontsize=12, pad=10)
        plt.tight_layout()

        # Convert chart to base64 string to embed in HTML
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#1a1a2e')
        buf.seek(0)
        chart_b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        return render_template('result.html',
            approved    = bool(prediction),
            probability = probability,
            reasons     = reasons,
            recommendations = recommendations,
            chart_b64   = chart_b64
        )

    except Exception as e:
        return render_template('index.html', error=str(e))


if __name__ == '__main__':
    app.run(debug=True)