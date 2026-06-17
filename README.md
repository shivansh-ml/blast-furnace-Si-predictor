# SI Predictor — Blast Furnace Web App

Extra Trees Regressor (R² = 98%) served as a Flask web app with MySQL storage.

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


## Status codes

| Status         | Meaning                          |
|----------------|----------------------------------|
| `normal`       | SI between 0.50 – 1.00           |
| `warning`      | SI between 0.40–0.50 or 1.00–1.20|
| `out_of_range` | SI < 0.40 or > 1.20              |