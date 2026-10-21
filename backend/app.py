"""
Flask API backend for Urban Flood Risk Analyzer
Serves prediction and data endpoints for the React frontend.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
MODELS_DIR = os.path.join(BASE, "models")
RESULTS_DIR = os.path.join(BASE, "results")
DATA_CANDIDATES = [
    os.path.join(ROOT, "data", "flood.csv"),
    r"C:\Users\PRISHA\Downloads\flood.csv",
]

app = Flask(__name__)
CORS(app)

rf_model = None
features = None
analysis_results = None
labeled_df = None
feature_bounds = None


def label_risk(p):
    if p < 0.47:
        return "LOW"
    if p <= 0.53:
        return "MEDIUM"
    return "HIGH"


def load_resources():
    global rf_model, features, analysis_results, labeled_df, feature_bounds
    rf_model = joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl"))
    features = joblib.load(os.path.join(MODELS_DIR, "features.pkl"))
    with open(os.path.join(RESULTS_DIR, "analysis_results.json")) as f:
        analysis_results = json.load(f)

    data_path = next((p for p in DATA_CANDIDATES if os.path.exists(p)), None)
    if data_path is None:
        raise FileNotFoundError("flood.csv not found in data/ or Downloads/")

    df = pd.read_csv(data_path)
    df["FloodRisk"] = df["FloodProbability"].apply(label_risk)
    labeled_df = df
    feature_bounds = {
        col: {"min": int(df[col].min()), "max": int(df[col].max()), "mean": round(float(df[col].mean()), 2)}
        for col in features
    }
    print(f"Resources loaded OK ({len(df)} rows from {data_path})")


def road_status(risk_level):
    return {"LOW": "Clear", "MEDIUM": "Caution", "HIGH": "Avoid"}.get(risk_level, "Unknown")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": rf_model is not None,
        "rows": int(len(labeled_df)) if labeled_df is not None else 0,
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify(analysis_results["dataset_summary"])


@app.route("/api/model-metrics", methods=["GET"])
def model_metrics():
    return jsonify({
        "metrics": analysis_results["model_metrics"],
        "production_model": "Random Forest",
        "target": "FloodRisk (LOW / MEDIUM / HIGH) from FloodProbability thresholds 0.47 and 0.53",
        "note": (
            "Logistic Regression reaches 100% because FloodRisk is a thresholded version of "
            "FloodProbability, and FloodProbability is a near-linear combination of the 20 factor scores. "
            "Random Forest is the saved production model used for live predictions."
        ),
    })


@app.route("/api/eda", methods=["GET"])
def eda():
    return jsonify(analysis_results["eda"])


@app.route("/api/statistical-tests", methods=["GET"])
def statistical_tests():
    return jsonify(analysis_results["statistical_tests"])


@app.route("/api/features", methods=["GET"])
def get_features():
    return jsonify({"features": features, "bounds": feature_bounds})


@app.route("/api/sample-row", methods=["GET"])
def sample_row():
    risk = request.args.get("risk")
    df = labeled_df
    if risk:
        df = df[df["FloodRisk"] == risk.upper()]
    if df.empty:
        return jsonify({"error": "No rows for that risk level"}), 404
    row = df.sample(1).iloc[0]
    payload = {f: int(row[f]) for f in features}
    payload["FloodProbability"] = float(row["FloodProbability"])
    payload["FloodRisk"] = row["FloodRisk"]
    return jsonify(payload)


@app.route("/api/samples", methods=["GET"])
def samples():
    n = min(int(request.args.get("n", 15)), 50)
    risk = request.args.get("risk")
    df = labeled_df
    if risk:
        df = df[df["FloodRisk"] == risk.upper()]
    sample = df.sample(min(n, len(df)))
    records = []
    for idx, row in sample.iterrows():
        records.append({
            "id": int(idx),
            "FloodProbability": round(float(row["FloodProbability"]), 4),
            "FloodRisk": row["FloodRisk"],
            "road_status": road_status(row["FloodRisk"]),
            "MonsoonIntensity": int(row["MonsoonIntensity"]),
            "Urbanization": int(row["Urbanization"]),
            "DrainageSystems": int(row["DrainageSystems"]),
            "DeterioratingInfrastructure": int(row["DeterioratingInfrastructure"]),
            "Landslides": int(row["Landslides"]),
            "Encroachments": int(row["Encroachments"]),
            "PopulationScore": int(row["PopulationScore"]),
            "Deforestation": int(row["Deforestation"]),
        })
    return jsonify(records)


@app.route("/api/scatter", methods=["GET"])
def scatter():
    x_col = request.args.get("x", "Urbanization")
    y_col = request.args.get("y", "DrainageSystems")
    n = min(int(request.args.get("n", 800)), 2000)
    if x_col not in labeled_df.columns or y_col not in labeled_df.columns:
        return jsonify({"error": "Unknown feature"}), 400
    sample = labeled_df.sample(min(n, len(labeled_df)), random_state=42)
    points = [
        {
            "x": int(row[x_col]),
            "y": int(row[y_col]),
            "prob": round(float(row["FloodProbability"]), 4),
            "risk": row["FloodRisk"],
        }
        for _, row in sample.iterrows()
    ]
    return jsonify({"x": x_col, "y": y_col, "points": points})


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    try:
        X = np.array([[float(body.get(f, 5)) for f in features]])
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    pred_class = rf_model.predict(X)[0]
    pred_proba = rf_model.predict_proba(X)[0]
    classes = rf_model.classes_
    proba_dict = {c: round(float(p) * 100, 2) for c, p in zip(classes, pred_proba)}
    class_order = ["LOW", "MEDIUM", "HIGH"]
    proba_list = [proba_dict.get(c, 0.0) for c in class_order]

    low_p = proba_dict.get("LOW", 0)
    med_p = proba_dict.get("MEDIUM", 0)
    high_p = proba_dict.get("HIGH", 0)
    risk_score = round(low_p * 0.165 + med_p * 0.5 + high_p * 0.835)

    return jsonify({
        "risk_level": pred_class,
        "risk_score": int(risk_score),
        "probabilities": {
            "LOW": proba_dict.get("LOW", 0),
            "MEDIUM": proba_dict.get("MEDIUM", 0),
            "HIGH": proba_dict.get("HIGH", 0),
        },
        "proba_list": proba_list,
        "classes": class_order,
        "road_status": road_status(pred_class),
    })


if __name__ == "__main__":
    load_resources()
    app.run(host="0.0.0.0", port=5000, debug=False)
