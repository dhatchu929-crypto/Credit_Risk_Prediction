
from http.server import BaseHTTPRequestHandler
import json
import joblib
import pandas as pd
from pathlib import Path

model_path = Path(__file__).resolve().parent.parent / "outputs" / "credit_risk_model.pkl"
model = joblib.load(model_path)

class handler(BaseHTTPRequestHandler):

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))

            columns = [
                "age",
                "monthly_income_inr",
                "employment_type",
                "credit_score",
                "loan_amount_inr",
                "loan_term_months",
                "existing_loans",
                "debt_to_income_ratio",
                "city_tier",
                "education"
            ]

            df = pd.DataFrame([data], columns=columns)
            probability = model.predict_proba(df)[0][1]

            result = {
                "risk": "High Risk" if probability >= 0.5 else "Low Risk",
                "probability": round(float(probability), 2)
            }

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
