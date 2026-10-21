# Urban Flood Risk & Road Safety Analyzer

End-to-end academic app: React dashboard + Flask API + Random Forest classifier trained on `flood.csv` (50,000 rows, 20 factor scores + `FloodProbability`).

Risk labels are derived from `FloodProbability`:

- **LOW** if p &lt; 0.47
- **MEDIUM** if 0.47 ≤ p ≤ 0.53
- **HIGH** if p &gt; 0.53

Live predictions use the saved Random Forest model (`backend/models/rf_model.pkl`). Metrics on the ML page are the real held-out test scores — they are not placeholders.

The dataset has **no coordinates or road names**. The Risk Map uses factor-space scatter plots, and Road Safety maps sampled rows’ flood labels to Clear / Caution / Avoid.

## Project layout

- `data/flood.csv` — source dataset
- `backend/` — Flask API, training script, saved model, analysis JSON
- `src/` — existing React UI, now wired to `/api`
- `public/graphs/` — EDA and evaluation charts from training
- `graphs/` and `results/` — original Person-2 EDA outputs (kept)

## Run (two terminals)

**1. Backend**

```bash
cd backend
pip install -r requirements.txt
python app.py
```

API: http://127.0.0.1:5000/api/health

Retrain (optional; overwrites the model and graphs):

```bash
python train_model.py
```

**2. Frontend**

```bash
npm install
npm run dev
```

Open the Vite URL (usually http://localhost:5173). The Vite proxy forwards `/api` to port 5000.

## What the model reports

Test-set metrics are stored in `backend/results/analysis_results.json` after training. Logistic Regression can reach 100% because `FloodRisk` is a threshold of `FloodProbability`, which is a near-linear combination of the 20 scores. Random Forest is the production predictor used by **Analyze Risk**.
