# 🔥 Calorie Burn Predictor

A machine-learning web app that predicts calories burnt during exercise from
seven biometric and activity features — gender, age, height, weight,
duration, heart rate, and body temperature — trained on 15,000 real workout
sessions.

Built for the MCA capstone project *"Calories Burnt Prediction Using Machine
Learning"* (Osmania University, Dept. of Computer Science).

## Project structure

```
calorie-burn-prediction/
├── app.py                  # Streamlit UI (Predict / Data Explorer / Model Lab / About)
├── model.py                # Data loading, preprocessing, training, prediction
├── requirements.txt
├── .streamlit/
│   └── config.toml         # Dark theme config
├── data/
│   └── calories.csv        # 15,000-row training dataset (must stay committed!)
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Deploy to Streamlit Community Cloud

1. Push this whole folder to a **public GitHub repo**, keeping the folder
   structure above intact (`data/calories.csv` must be committed — see the
   note in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Point it at your repo, branch, and set **Main file path** to `app.py`.
4. Deploy. First build takes ~2-3 minutes (installing xgboost/scikit-learn).

### If you hit `FileNotFoundError: data/calories.csv` again

This almost always means the CSV never made it into the GitHub repo. Check:

```bash
git ls-files | grep calories.csv
```

If nothing prints:
- Check `.gitignore` doesn't contain a rule like `data/` or `*.csv`.
- Confirm the file isn't tracked via Git LFS (Streamlit Cloud can fail to
  pull LFS objects) — run `git lfs ls-files` to check.
- Re-add and commit it explicitly: `git add -f data/calories.csv`.

`model.py` resolves the CSV path relative to its own file location
(`os.path.dirname(os.path.abspath(__file__))`), not the process's working
directory, so path issues caused by *how* Streamlit Cloud launches the app
are already handled — the remaining failure mode is almost always the file
missing from the repo itself.

## Model

Five regressors are trained and compared on an 80/20 split with
`StandardScaler`-normalised features: Linear Regression, Ridge, Lasso,
Random Forest, and XGBoost. The best model on held-out validation R² is
selected automatically at runtime and used for all live predictions — no
hardcoded model choice, so re-running on updated data re-selects the winner.

## Credits

- **Author:** Anushka Nizamabad (Roll No. 1010-23-862-005)
- **Guide:** Dr. G. Vamshi Krishna, Asst. Professor, Dept. of Computer Science, UPGCS (OU)
- **Dataset:** 15,000-row calories/exercise dataset (Kaggle-style: `calories.csv`)
