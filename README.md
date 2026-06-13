# SI Predictor — Blast Furnace Web App

Extra Trees Regressor (R² = 87.35%) served as a Flask web app with MySQL storage.

---

## Project Structure

```
si_predictor/
├── app.py                    ← Flask backend
├── requirements.txt
├── setup_db.sql              ← Run once to create MySQL tables
├── si_prediction_model.pkl   ← Copy here from rf4.py output  ⬅ REQUIRED
└── templates/
    ├── index.html            ← Landing page
    ├── predict.html          ← Prediction form
```

---

## Step-by-Step Setup

### 1 — Copy your model file

After running your ML Model, export the model into a pkl file and copy the saved model into this folder:
```
cp si_prediction_model.pkl  /path/to/si_predictor/
```

### 2 — Create the MySQL table

Open your MySQL client and run:
```sql
source setup_db.sql
```
Or from the terminal:
```bash
mysql -u root -p < setup_db.sql
```

### 3 — Edit DB credentials in app.py

Open `app.py` and update the `DB_CONFIG` block (lines 11–18):
```python
DB_CONFIG = {
    "host":     "localhost",       # your MySQL host
    "user":     "root",            # your MySQL user
    "password": "your_password",   # ← change this
    "database": "blast_furnace",
    "port":     3306,
}
```

Or set environment variables (recommended for production):
```bash
export DB_HOST=localhost
export DB_USER=root
export DB_PASSWORD=your_password
export DB_NAME=blast_furnace
```
### 4 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### 5 — Run the app

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## Pages

| URL            | What it does                                      |
|----------------|---------------------------------------------------|
| `/`            | Landing page — model metrics & feature importance |
| `/predict`     | Input form → real-time SI prediction              |

## API Endpoints

| Endpoint          | Method | Description                          |
|-------------------|--------|--------------------------------------|
| `/predict`        | POST   | JSON body → `{"prediction": 0.72, …}`|
| `/api/db-records` | GET    | Last 50 rows from blast_furnace_data |

### Example POST to /predict
```json
{
  "HB_PRES": 450, "HB_VOL": 1250, "HB_TEMP": 1060,
  "TEMP_HM": 1490, "FUEL_INJ": 155, "O2_FLOW": 7800,
  "STEAM_FLOW": 4800, "HEAT_FLUX": 118, "TOP_PRES": 2.4,
  "BURDEN_RES": 0.82, "RAFT": 2140, "ETACO": 0.476,
  "COKERATE": 345, "PCIRATE": 148,
  "M40COB6ANAL": 82.5, "M10COB6ANAL": 6.8,
  "CSRCOB6ANAL": 62, "CRICOB6ANAL": 24.5,
  "ASHCOB6ANAL": 10.2, "VMPCIANAL": 17.8,
  "IMPCIANAL": 54, "ASHPCIANAL": 11.5,
  "FE_PELLETANAL": 65.2, "SIO2_PELLETANAL": 3.4,
  "AL2O3_PELLETANAL": 0.75,
  "FE_SINTERANAL": 56.3, "SIO2_SINTERANAL": 5.1,
  "CAO_SINTERANAL": 9.8, "MGO_SINTERANAL": 1.75,
  "LIME_SINTERANAL": 1.92, "BSTY_SINTERANAL": 2.08,
  "FE_OREANAL": 61.8, "SIO2_OREANAL": 4.2,
  "AL2O3_OREANAL": 2.3,
  "actual_si": 0.72
}
```

### Response
```json
{
  "prediction": 0.7183,
  "status": "normal",
  "actual": 0.72,
  "error": 0.0017,
  "correct": true
}
```

---

## Status codes

| Status         | Meaning                          |
|----------------|----------------------------------|
| `normal`       | SI between 0.50 – 1.00           |
| `warning`      | SI between 0.40–0.50 or 1.00–1.20|
| `out_of_range` | SI < 0.40 or > 1.20              |