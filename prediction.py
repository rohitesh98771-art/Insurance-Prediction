"""
Insurance Calculator (learns from your CSV)
-------------------------------------------
1. Run the program.
2. Give it the path of your CSV file (or press Enter to open a file picker).
3. It trains a model on the data and reports how accurate it is.
4. Enter new data and it predicts the insurance charges.

Install once (VS Code terminal):
    pip install pandas scikit-learn joblib

Run:
    python insurance_ml_calculator.py
"""

import os
import sys

try:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import cross_val_predict, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError:
    print("Missing libraries. Run this in the VS Code terminal first:")
    print("    pip install pandas scikit-learn joblib")
    sys.exit(1)

CURRENCY = "$"  # change to "₹", "£", "€" ...
MODEL_FILE = "insurance_model.joblib"
RANDOM_STATE = 42


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def money(x):
    return f"{CURRENCY}{x:,.2f}"


def pick_file_with_dialog():
    """Open a file-picker window (works on Windows/Mac/most Linux)."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Select your insurance CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
        return path
    except Exception:
        return ""


def ask_for_csv():
    """Keep asking until we get a readable CSV."""
    while True:
        raw = input(
            "\nEnter the path of your CSV file\n"
            "(or press Enter to browse, or type 'q' to quit): "
        ).strip().strip('"').strip("'")

        if raw.lower() == "q":
            sys.exit(0)
        if raw == "":
            raw = pick_file_with_dialog()
            if not raw:
                print("  No file selected.")
                continue
        raw = os.path.expanduser(raw)
        if not os.path.isfile(raw):
            print(f"  File not found: {raw}")
            continue
        try:
            df = pd.read_csv(raw)
        except Exception as e:
            print(f"  Could not read the file as CSV: {e}")
            continue
        if df.shape[1] < 2 or len(df) < 30:
            print("  The file needs at least 2 columns and 30 rows.")
            continue
        return raw, df


def choose_target(df):
    """Use 'charges' if present, otherwise let the user pick."""
    for name in df.columns:
        if name.lower() in ("charges", "charge", "premium", "cost", "expenses"):
            return name
    print("\nColumns found:", ", ".join(df.columns))
    while True:
        name = input("Which column should be predicted (the price/charges)? ").strip()
        if name in df.columns and pd.api.types.is_numeric_dtype(df[name]):
            return name
        print("  Enter an exact numeric column name.")


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def build_pipeline(model, num_cols, cat_cols):
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return Pipeline([("prep", pre), ("model", model)])


def train(df, target):
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    X = df.drop(columns=[target])
    y = df[target]

    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
    }

    print("\nLearning from your data...")
    print(f"  Rows used: {len(df)}   Features: {', '.join(X.columns)}")
    print(f"\n  {'Model':<20}{'R2 (test)':>10}{'Avg error':>14}")
    best_name, best_pipe, best_r2 = None, None, -np.inf
    for name, model in candidates.items():
        pipe = build_pipeline(model, num_cols, cat_cols)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        print(f"  {name:<20}{r2:>10.3f}{money(mae):>14}")
        if r2 > best_r2:
            best_name, best_pipe, best_r2 = name, pipe, r2

    print(f"\n  Best model: {best_name} (R2 = {best_r2:.3f})")

    # Estimate typical error range from cross-validated predictions
    cv_pred = cross_val_predict(best_pipe, X, y, cv=5)
    ratio = (y.values - cv_pred) / np.maximum(cv_pred, 1.0)
    low_q, high_q = np.percentile(ratio, [10, 90])

    # Final model learns from ALL rows
    best_pipe.fit(X, y)

    
    meta = {
        "target": target,
        "features": list(X.columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_range": {c: (float(X[c].min()), float(X[c].max())) for c in num_cols},
        "num_is_int": {c: bool(pd.api.types.is_integer_dtype(X[c])) for c in num_cols},
        "cat_values": {c: sorted(X[c].astype(str).unique().tolist()) for c in cat_cols},
        "ratio_band": (float(low_q), float(high_q)),
        "model_name": best_name,
        "r2": float(best_r2),
    }
    return best_pipe, meta


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
def ask_number(col, lo, hi, is_int):
    while True:
        raw = input(f"  {col} (data range {lo:g} - {hi:g}): ").strip().replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            print("    Please enter a number.")
            continue
        if is_int and val != int(val):
            print("    Please enter a whole number.")
            continue
        if val < 0:
            print("    Value cannot be negative.")
            continue
        if val < lo or val > hi:
            print(
                f"    Note: {val:g} is outside the data the model learned from "
                f"({lo:g}-{hi:g}); the estimate may be less reliable."
            )
        return int(val) if is_int else val

def ask_category(col, options):
    lowered = {o.lower(): o for o in options}
    while True:
        raw = input(f"  {col} ({' / '.join(options)}): ").strip().lower()
        if raw in lowered:
            return lowered[raw]
        # allow y/n shortcuts for yes/no columns
        if set(lowered) == {"yes", "no"} and raw in ("y", "n"):
            return lowered["yes" if raw == "y" else "no"]
        print(f"    Choose one of: {', '.join(options)}")

def predict_once(pipe, meta):
    print("\nEnter the new person's details:")
    row = {}
    for col in meta["features"]:
        if col in meta["cat_cols"]:
            row[col] = ask_category(col, meta["cat_values"][col])
        else:
            lo, hi = meta["num_range"][col]
            row[col] = ask_number(col, lo, hi, meta["num_is_int"][col])

    pred = float(pipe.predict(pd.DataFrame([row]))[0])
    pred = max(pred, 0.0)
    lo_r, hi_r = meta["ratio_band"]
    low, high = pred * (1 + lo_r), pred * (1 + hi_r)

    print("\n" + "=" * 50)
    print(f"  Predicted {meta['target']} (per year): {money(pred)}")
    print(f"  Likely range (80% of cases):      {money(low)} - {money(high)}")
    print(f"  Approx. per month:                {money(pred / 12)}")
    print("=" * 50)
    return row, pred

def load_or_train():
    if os.path.isfile(MODEL_FILE):
        ans = input(
            f"\nA previously trained model was found ({MODEL_FILE}).\n"
            "Use it? (y = use saved model, n = train from a CSV): "
        ).strip().lower()
        if ans in ("y", "yes"):
            try:
                saved = joblib.load(MODEL_FILE)
                return saved["pipe"], saved["meta"]
            except Exception as e:
                print(f"  Could not load saved model ({e}). Training a new one.")

    path, df = ask_for_csv()
    target = choose_target(df)
    pipe, meta = train(df, target)
    try:
        joblib.dump({"pipe": pipe, "meta": meta}, MODEL_FILE)
        print(f"  Model saved to {MODEL_FILE}")
    except Exception:
        pass
    return pipe, meta

def main():
    print("=" * 50)
    print("     INSURANCE CALCULATOR - learns from your CSV")
    print("   (estimates only - not an official quote)")
    print("=" * 50)

    pipe, meta = load_or_train()
    history = []

    while True:
        print("\nMenu:  1) New prediction   2) Show history   3) Load a different CSV   0) Exit")
        choice = input("Choose: ").strip()
        if choice == "1":
            row, pred = predict_once(pipe, meta)
            history.append((row, pred))
        elif choice == "2":
            if not history:
                print("  No predictions yet.")
            for i, (row, pred) in enumerate(history, 1):
                details = ", ".join(f"{k}={v}" for k, v in row.items())
                print(f"  {i}. {money(pred)}  <-  {details}")
        elif choice == "3":
            path, df = ask_for_csv()
            target = choose_target(df)
            pipe, meta = train(df, target)
            try:
                joblib.dump({"pipe": pipe, "meta": meta}, MODEL_FILE)
            except Exception:
                pass
            history.clear()
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("  Invalid choice.")
if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")