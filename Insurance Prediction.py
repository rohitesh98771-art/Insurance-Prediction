"""
Insurance Calculator - learns from your CSV

Install once:  pip install pandas scikit-learn joblib matplotlib
Run:           python insurance_ml_calculator.py
"""

import os
import sys

try:
    import joblib
    import numpy as np
    import pandas as pd

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import cross_val_predict, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
except ImportError:
    print("Missing libraries. Run this in the VS Code terminal first:")
    print("    pip install pandas scikit-learn joblib matplotlib")
    sys.exit(1)

# Charts are optional: the program still works if matplotlib is missing
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

CURRENCY = "$"
MODEL_FILE = "insurance_model.joblib"
DASHBOARD_FILE = "insurance_dashboard.png"
PREDICTION_FILE = "insurance_prediction.png"
RANDOM_STATE = 42

BLUE, GREEN, ORANGE, GREY = "#3b82f6", "#10b981", "#f59e0b", "#9ca3af"


def money(x):
    return f"{CURRENCY}{x:,.2f}"


# ----------------------------------------------------------------------
# Loading the CSV
# ----------------------------------------------------------------------

def pick_file_with_dialog():
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
    # Keep asking until a readable CSV is given
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
    # Use the price column automatically if its name is recognised
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
# Data cleaning
# ----------------------------------------------------------------------

def clean_data(df, target):
    start_rows = len(df)
    df = df.copy()

    # Tidy text columns: " Male", "MALE" and "male" become the same value
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace({"": np.nan, "nan": np.nan, "none": np.nan})

    # Report missing values, then drop rows that have any
    missing = df.isna().sum()
    missing = missing[missing > 0]
    df = df.dropna()

    # Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    duplicates = before - len(df)

    # A price of zero or less is not a valid training example
    df = df[df[target] > 0].reset_index(drop=True)

    print("\nData cleaning report")
    print(f"  Rows loaded:          {start_rows}")
    print(f"  Missing values:       {'none' if missing.empty else missing.to_dict()}")
    print(f"  Duplicate rows:       {duplicates} removed")
    print(f"  Rows used to learn:   {len(df)}")

    return df


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def build_pipeline(model, num_cols, cat_cols):
    # Numbers are scaled, text columns are converted to 0/1 columns
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([("prep", pre), ("model", model)])


def train(df, target):
    df = clean_data(df, target)

    X = df.drop(columns=[target])
    y = df[target]

    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # 80% of the rows are used to learn, 20% are kept back to test accuracy
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
    print(f"\n  {'Model':<20}{'R2 (test)':>10}{'Avg error':>14}")

    scores = {}
    best_name, best_pipe, best_r2 = None, None, -np.inf

    # Train every model and keep the most accurate one
    for name, model in candidates.items():
        pipe = build_pipeline(model, num_cols, cat_cols)
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        r2 = r2_score(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        scores[name] = r2

        print(f"  {name:<20}{r2:>10.3f}{money(mae):>14}")

        if r2 > best_r2:
            best_name, best_pipe, best_r2 = name, pipe, r2

    print(f"\n  Best model: {best_name} (R2 = {best_r2:.3f})")

    # Which inputs matter most (measured on the unseen test rows)
    importance = permutation_importance(
        best_pipe, X_test, y_test, n_repeats=5, random_state=RANDOM_STATE
    )

    # Typical error size, used later for the "likely range"
    cv_pred = cross_val_predict(best_pipe, X, y, cv=5)
    ratio = (y.values - cv_pred) / np.maximum(cv_pred, 1.0)
    low_q, high_q = np.percentile(ratio, [10, 90])

    # The final model learns from all rows
    best_pipe.fit(X, y)

    meta = {
        "target": target,
        "features": list(X.columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "num_range": {c: (float(X[c].min()), float(X[c].max())) for c in num_cols},
        "num_is_int": {c: bool(pd.api.types.is_integer_dtype(X[c])) for c in num_cols},
        "cat_values": {c: sorted(X[c].unique().tolist()) for c in cat_cols},
        "ratio_band": (float(low_q), float(high_q)),
        "target_mean": float(y.mean()),
    }

    # Data used only for drawing the charts
    report = {
        "scores": scores,
        "best_name": best_name,
        "y_test": y_test.values,
        "y_pred": best_pipe.predict(X_test),
        "importance": dict(zip(X.columns, importance.importances_mean)),
        "df": df,
        "target": target,
        "cat_cols": cat_cols,
    }

    return best_pipe, meta, report


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------

def open_image(path):
    # Open the saved PNG with the computer's default image viewer
    import subprocess

    full = os.path.abspath(path)

    try:
        if sys.platform.startswith("win"):
            os.startfile(full)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", full])
        else:
            subprocess.Popen(["xdg-open", full])
    except Exception:
        print(f"  Open this file manually to see the chart: {full}")


def display_figure(path):
    # Show a chart window; if this Python cannot open windows, open the PNG instead
    backend = plt.get_backend().lower()

    if backend in ("agg", "pdf", "svg", "ps", "cairo", "template"):
        print(f"  No chart window available here, opening {path} instead.")
        plt.close("all")
        open_image(path)
    else:
        plt.show()


def show_dashboard(report):
    if not HAS_PLOT:
        print("  Charts skipped: matplotlib is not installed (pip install matplotlib).")
        return

    target = report["target"]
    df = report["df"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("What the model learned from your data", fontsize=15, fontweight="bold")

    # 1) Accuracy of each model
    ax = axes[0, 0]
    names = list(report["scores"])
    vals = list(report["scores"].values())
    colors = [GREEN if n == report["best_name"] else GREY for n in names]
    ax.bar(names, vals, color=colors)
    ax.set_title("Model accuracy (R2, higher is better)")
    ax.set_ylim(0, 1)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

    # 2) Predicted vs actual on unseen test rows
    ax = axes[0, 1]
    ax.scatter(report["y_test"], report["y_pred"], s=14, alpha=0.6, color=BLUE)
    top = max(report["y_test"].max(), report["y_pred"].max())
    ax.plot([0, top], [0, top], "--", color=ORANGE, label="Perfect prediction")
    ax.set_title("Predicted vs actual (test data)")
    ax.set_xlabel(f"Actual {target}")
    ax.set_ylabel(f"Predicted {target}")
    ax.legend()

    # 3) Which inputs affect the price most
    ax = axes[1, 0]
    imp = sorted(report["importance"].items(), key=lambda kv: kv[1])
    ax.barh([k for k, _ in imp], [v for _, v in imp], color=BLUE)
    ax.set_title("Most important inputs")
    ax.set_xlabel("Importance (drop in accuracy if removed)")

    # 4) Average price for the categorical column with the biggest gap
    ax = axes[1, 1]
    cat_cols = report["cat_cols"]
    if cat_cols:
        spread = {c: df.groupby(c)[target].mean().agg(lambda s: s.max() - s.min()) for c in cat_cols}
        col = max(spread, key=spread.get)
        means = df.groupby(col)[target].mean().sort_values()
        ax.bar(means.index, means.values, color=ORANGE)
        ax.set_title(f"Average {target} by {col}")
        for i, v in enumerate(means.values):
            ax.text(i, v, money(v), ha="center", va="bottom")
    else:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(DASHBOARD_FILE, dpi=130)
    print(f"  Charts saved as {DASHBOARD_FILE}. Close the chart window to continue.")
    display_figure(DASHBOARD_FILE)


def show_prediction_chart(pred, low, high, avg, target):
    if not HAS_PLOT:
        return

    labels = ["Low estimate", "Predicted", "High estimate", "Dataset average"]
    values = [low, pred, high, avg]
    colors = [GREY, GREEN, GREY, ORANGE]

    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, values, color=colors)
    plt.title(f"Predicted {target} compared with the dataset average")

    for i, v in enumerate(values):
        plt.text(i, v, money(v), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(PREDICTION_FILE, dpi=130)
    display_figure(PREDICTION_FILE)


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

        # Values outside the learned range are allowed but flagged
        if val < lo or val > hi:
            print(f"    Note: outside the learned range ({lo:g}-{hi:g}); estimate may be less reliable.")

        return int(val) if is_int else val


def ask_category(col, options):
    lowered = {o.lower(): o for o in options}

    while True:
        raw = input(f"  {col} ({' / '.join(options)}): ").strip().lower()

        if raw in lowered:
            return lowered[raw]

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

    pred = max(float(pipe.predict(pd.DataFrame([row]))[0]), 0.0)

    lo_r, hi_r = meta["ratio_band"]
    low, high = pred * (1 + lo_r), pred * (1 + hi_r)

    print("\n" + "=" * 50)
    print(f"  Predicted {meta['target']} (per year): {money(pred)}")
    print(f"  Likely range (80% of cases):      {money(low)} - {money(high)}")
    print(f"  Approx. per month:                {money(pred / 12)}")
    print("=" * 50)

    if HAS_PLOT and input("\nShow chart? (y/n): ").strip().lower() in ("y", "yes"):
        show_prediction_chart(pred, low, high, meta["target_mean"], meta["target"])

    return row, pred


# ----------------------------------------------------------------------
# Main program
# ----------------------------------------------------------------------

def train_from_csv():
    path, df = ask_for_csv()
    target = choose_target(df)
    pipe, meta, report = train(df, target)

    # Save the trained model so it can be reused next time
    try:
        joblib.dump({"pipe": pipe, "meta": meta, "report": report}, MODEL_FILE)
        print(f"  Model saved to {MODEL_FILE}")
    except Exception:
        pass

    show_dashboard(report)
    return pipe, meta, report


def load_or_train():
    if os.path.isfile(MODEL_FILE):
        ans = input(
            f"\nA previously trained model was found ({MODEL_FILE}).\n"
            "Use it? (y = use saved model, n = train from a CSV): "
        ).strip().lower()

        if ans in ("y", "yes"):
            try:
                saved = joblib.load(MODEL_FILE)
                return saved["pipe"], saved["meta"], saved.get("report")
            except Exception as e:
                print(f"  Could not load saved model ({e}). Training a new one.")

    return train_from_csv()


def main():
    print("=" * 50)
    print("     INSURANCE CALCULATOR - learns from your CSV")
    print("   (estimates only - not an official quote)")
    print("=" * 50)

    if HAS_PLOT:
        print(f"  Charts: ON (backend: {plt.get_backend()})")
    else:
        print("  Charts: OFF - matplotlib is not installed.")
        print("  Run  pip install matplotlib  and start the program again.")

    pipe, meta, report = load_or_train()
    history = []

    while True:
        print("\nMenu:  1) New prediction   2) Show history   3) Load a different CSV   4) Show charts   0) Exit")
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
            pipe, meta, report = train_from_csv()
            history.clear()

        elif choice == "4":
            if report:
                show_dashboard(report)
            else:
                print("  No chart data in the saved model. Choose 3 and load the CSV again.")

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
