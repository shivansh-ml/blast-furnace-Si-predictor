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
STEAM_ZERO_MODEL = "si_prediction_model_steam_zero.pkl"
STEAM_NONZERO_MODEL = "si_prediction_steam_flow_nonzero_model.pkl"

steam_zero_package = None
steam_nonzero_package = None

def load_models():

    global steam_zero_package
    global steam_nonzero_package

    steam_zero_package = joblib.load(
        STEAM_ZERO_MODEL
    )

    steam_nonzero_package = joblib.load(
        STEAM_NONZERO_MODEL
    )

    print(
        f"[OK] Steam Zero Model Loaded : "
        f"{len(steam_zero_package['feature_columns'])} features"
    )

    print(
        f"[OK] Steam Non-Zero Model Loaded : "
        f"{len(steam_nonzero_package['feature_columns'])} features"
    )

load_models()

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


# ─── FEATURE ENGINEERING ────────────────────────────────────
def engineer_features(raw, feature_columns):

    X = pd.DataFrame([raw])

    for col in feature_columns:

        if col not in X.columns:
            X[col] = np.nan

    return X[feature_columns]

def run_prediction_on_row(row: dict, row_num: int = None):
    raw = {
        k: (float(v) if v is not None else np.nan)
        for k, v in row.items()
        if k not in LEAKAGE_COLS and k != "SI_PRED"
        and k not in ("SL_NO","SI_PRED_DATETIME","SI_PREV","SI_PREV_DATETIME",
                      "BASICITY_SLAG","SILICA_LOAD","RUNMODE",
                      "SIPREV_CASTNO","CURR_CASTNO")
    }
    steam_flow = raw.get("STEAM_FLOW", 0)

    if steam_flow == 0:
        package = steam_zero_package
    else:
        package = steam_nonzero_package

    X_input = engineer_features(
        raw,
        package["feature_columns"]
    )

    X_scaled = pd.DataFrame(
        package["scaler"].transform(X_input),
        columns=package["feature_columns"]
    )

    pred = round(
        float(
            package["model"].predict(X_scaled)[0]
        ),
        4
    )

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
        "etaco": raw.get("ETACO"),
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
    if (steam_zero_package is None or steam_nonzero_package is None):
        return jsonify(
            {"error": "Models not loaded"}
        ), 500

    try:
        raw = {k: float(v) for k, v in data.items() if k != "actual_si"}
        actual_si = float(data["actual_si"]) if data.get("actual_si") else None

        steam_flow = raw.get("STEAM_FLOW", 0)

        if steam_flow == 0:
            package = steam_zero_package
        else:
            package = steam_nonzero_package

        X_input = engineer_features(
            raw,
            package["feature_columns"]
        )

        X_scaled = pd.DataFrame(
        package["scaler"].transform(X_input),
        columns=package["feature_columns"])

        pred = round(
            float(
                    package["model"].predict(X_scaled)[0]),4)

        status = "normal"
        if pred < 0.4 or pred > 1.2:
            status = "out_of_range"
        elif pred < 0.5 or pred > 1.0:
            status = "warning"

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