from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import pymysql
import os
from datetime import datetime

app = Flask(__name__)

# ─── DB CONFIG ──────────────────────────────────────────────
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "localhost"),
    "user":     os.getenv("DB_USER",     "root"),
    "password": os.getenv("DB_PASSWORD", "Shivansh@123"),
    "database": os.getenv("DB_NAME",     "hotmetal"),
    "port":     int(os.getenv("DB_PORT", 3306)),
    "cursorclass": pymysql.cursors.DictCursor
}

LEAKAGE_COLS = {
    "C_HM","MN_HM","S_HM","P_HM","TI_HM",
    "SIO2_SLAG","FEO_SLAG","AL2O3_SLAG","CAO_SLAG",
    "MGO_SLAG","MNO_SLAG","TIO2_SLAG","K2O_SLAG"
}

# ─── LOAD MODEL ─────────────────────────────────────────────
MODEL_PATH = "si_prediction_model.pkl"
model_package = None

def load_model():
    global model_package
    try:
        model_package = joblib.load(MODEL_PATH)
        print(f"[OK] Model loaded — {len(model_package['feature_columns'])} features")
    except FileNotFoundError:
        print(f"[WARN] {MODEL_PATH} not found — prediction disabled")

load_model()

# ─── DB HELPERS ─────────────────────────────────────────────
def get_db():
    return pymysql.connect(**DB_CONFIG)

def get_recent_records(limit=50):
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM bf5_data LIMIT %s", (limit,))
            rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"[DB] get_recent_records error: {e}")
        return []

def save_prediction(features: dict, predicted_si: float, actual_si: float = None):
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO predictions
                   (predicted_si, actual_si, hb_pres, temp_hm, fuel_inj,
                    o2_flow, heat_flux, raft, etaco, cokerate, pcirate,
                    created_at)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    predicted_si, actual_si,
                    features.get("HB_PRES"), features.get("TEMP_HM"),
                    features.get("FUEL_INJ"), features.get("O2_FLOW"),
                    features.get("HEAT_FLUX"), features.get("RAFT"),
                    features.get("ETACO"), features.get("COKERATE"),
                    features.get("PCIRATE"), datetime.now()
                )
            )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] save_prediction error: {e}")

def get_prediction_history(limit=100):
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, predicted_si, actual_si, hb_pres, temp_hm,
                          fuel_inj, created_at
                   FROM predictions ORDER BY created_at DESC LIMIT %s""",
                (limit,)
            )
            rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        return []

def get_dashboard_stats():
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS total FROM predictions")
            total = cur.fetchone()["total"]
            cur.execute(
                """SELECT AVG(predicted_si) AS avg_si,
                          MIN(predicted_si) AS min_si,
                          MAX(predicted_si) AS max_si
                   FROM predictions"""
            )
            stats = cur.fetchone()
            cur.execute(
                """SELECT DATE(created_at) AS day, COUNT(*) AS cnt,
                          AVG(predicted_si) AS avg_si
                   FROM predictions GROUP BY day ORDER BY day DESC LIMIT 14"""
            )
            trend = cur.fetchall()
        conn.close()
        return {"total": total, "stats": stats, "trend": trend}
    except Exception as e:
        return {"total": 0, "stats": {}, "trend": []}

# ─── FEATURE ENGINEERING ────────────────────────────────────
def engineer_features(raw: dict) -> pd.DataFrame:
    X = pd.DataFrame([raw])
    X["STEAM_O2_RATIO"]  = X["STEAM_FLOW"] / (X["O2_FLOW"] + 1e-6)
    X["HEAT_BURDEN"]     = X["HEAT_FLUX"] * X["BURDEN_RES"]
    X["RAFT_ETACO"]      = X["RAFT"] * X["ETACO"]
    X["PELLET_FE_SIO2"]  = X["FE_PELLETANAL"] / (X["SIO2_PELLETANAL"] + 1e-3)
    X["SINTER_FE_SIO2"]  = X["FE_SINTERANAL"] / (X["SIO2_SINTERANAL"] + 1e-3)
    X["FUEL_O2"]         = X["FUEL_INJ"] * X["O2_FLOW"]
    X["TEMP_ETACO"]      = X["TEMP_HM"] * X["ETACO"]
    X["COKE_PCI_TOTAL"]  = X["COKERATE"] + X["PCIRATE"]
    X["M40_M10_RATIO"]   = X["M40COB6ANAL"] / (X["M10COB6ANAL"] + 1e-3)
    X["CSR_CRI_RATIO"]   = X["CSRCOB6ANAL"] / (X["CRICOB6ANAL"] + 1e-3)
    X["HB_PRES_VOL"]     = X["HB_PRES"] * X["HB_VOL"]
    X["HB_PRES_TEMP"]    = X["HB_PRES"] * X["HB_TEMP"]
    X["RAFT_HB_PRES"]    = X["RAFT"] * X["HB_PRES"]
    X["HEAT_O2"]         = X["HEAT_FLUX"] * X["O2_FLOW"]
    X["TOP_PRES_BURDEN"] = X["TOP_PRES"] * X["BURDEN_RES"]
    X["O2_ETACO"]        = X["O2_FLOW"] * X["ETACO"]
    X["FUEL_ETACO"]      = X["FUEL_INJ"] * X["ETACO"]
    X["HEAT_TOP"]        = X["HEAT_FLUX"] * X["TOP_PRES"]
    X["ETACO_SQ"]        = X["ETACO"] ** 2
    X["HEAT_FLUX_SQ"]    = X["HEAT_FLUX"] ** 2
    # If any expected feature is missing (e.g. NULL in DB), add it as NaN
    for col in model_package["feature_columns"]:
        if col not in X.columns:
            X[col] = np.nan

    return X[model_package["feature_columns"]]

def run_prediction_on_row(row: dict, row_num: int = None):
    raw = {
        k: (float(v) if v is not None else np.nan)
        for k, v in row.items()
        if k not in LEAKAGE_COLS and k != "SI_PRED"
        and k not in ("SL_NO","SI_PRED_DATETIME","SI_PREV","SI_PREV_DATETIME",
                      "BASICITY_SLAG","SILICA_LOAD","RUNMODE",
                      "SIPREV_CASTNO","CURR_CASTNO")
    }
    X_input  = engineer_features(raw)
    X_scaled = pd.DataFrame(
        model_package["scaler"].transform(X_input),
        columns=model_package["feature_columns"]
    )
    pred = round(float(model_package["model"].predict(X_scaled)[0]), 4)

    status = "normal"
    if pred < 0.4 or pred > 1.2:
        status = "out_of_range"
    elif pred < 0.5 or pred > 1.0:
        status = "warning"

    result = {
        "row_num": row_num,
        "prediction": pred, "status": status,
        "hb_pres": raw.get("HB_PRES"), "temp_hm": raw.get("TEMP_HM"),
        "fuel_inj": raw.get("FUEL_INJ"), "raft": raw.get("RAFT"),
        "etaco": raw.get("ETACO"), "cokerate": raw.get("COKERATE"),
        "pcirate": raw.get("PCIRATE"),
    }

    actual = row.get("SI_PRED")
    if actual is not None:
        try:
            actual = round(float(actual), 4)
            error  = round(abs(pred - actual), 4)
            result["actual"]  = actual
            result["error"]   = error
            result["correct"] = error <= 0.05
        except (TypeError, ValueError):
            pass

    return result, raw

# ─── ROUTES ─────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    data = request.get_json(force=True)
    if model_package is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        raw = {k: float(v) for k, v in data.items() if k != "actual_si"}
        actual_si = float(data["actual_si"]) if data.get("actual_si") else None

        X_input = engineer_features(raw)
        X_scaled = pd.DataFrame(
            model_package["scaler"].transform(X_input),
            columns=model_package["feature_columns"]
        )
        pred = round(float(model_package["model"].predict(X_scaled)[0]), 4)

        status = "normal"
        if pred < 0.4 or pred > 1.2:
            status = "out_of_range"
        elif pred < 0.5 or pred > 1.0:
            status = "warning"

        save_prediction(raw, pred, actual_si)

        result = {"prediction": pred, "status": status}
        if actual_si is not None:
            error = abs(pred - actual_si)
            result["actual"]  = actual_si
            result["error"]   = round(error, 4)
            result["correct"] = error <= 0.05

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/last-db-record")
def api_last_db_record():
    """Fetch one row from bf5_data to auto-fill the predict form."""
    try:
        conn = get_db()
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM bf5_data ORDER BY SL_NO DESC LIMIT 1")
            row = cur.fetchone()
        conn.close()

        if row is None:
            return jsonify({"error": "No records found in bf5_data"}), 404

        clean = {}
        for k, v in row.items():
            if v is None:
                clean[k] = None
            else:
                try:
                    clean[k] = float(v)
                except (TypeError, ValueError):
                    clean[k] = str(v)

        return jsonify(clean)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)