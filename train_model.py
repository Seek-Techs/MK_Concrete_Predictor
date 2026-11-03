from pathlib import Path
import hashlib, json, platform
import joblib
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Concrete_Modelling_Dataset_Step4A.xlsx"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)
FEATURES = ["MK_replacement_pct", "curing_age_days"]
TARGET = "mean_MPa"

def build_model():
    return Pipeline([
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()),
        ("regression", LinearRegression()),
    ])

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()

def create_figures(df, test, predictions):
    """Create reproducible EDA and frozen-model holdout diagnostic figures."""
    # 1. Mean strength against curing age, one line per MK replacement level.
    fig, ax = plt.subplots(figsize=(9, 6))
    for mk, group in df.groupby("MK_replacement_pct", sort=True):
        group = group.sort_values("curing_age_days")
        ax.plot(group["curing_age_days"], group[TARGET], marker="o", label=f"{mk:g}% MK")
    ax.set(title="Mean Compressive Strength against Curing Age",
           xlabel="Curing age (days)", ylabel="Mean compressive strength (MPa)")
    ax.set_xticks(sorted(df["curing_age_days"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Metakaolin replacement", ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "01_strength_vs_curing_age.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Mean strength against MK replacement, one line per curing age.
    fig, ax = plt.subplots(figsize=(9, 6))
    for age, group in df.groupby("curing_age_days", sort=True):
        group = group.sort_values("MK_replacement_pct")
        ax.plot(group["MK_replacement_pct"], group[TARGET], marker="o", label=f"{age} days")
    ax.set(title="Mean Compressive Strength against Metakaolin Replacement",
           xlabel="Metakaolin replacement (%)", ylabel="Mean compressive strength (MPa)")
    ax.set_xticks(sorted(df["MK_replacement_pct"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend(title="Curing age")
    fig.tight_layout()
    fig.savefig(FIGURES / "02_strength_vs_mk_replacement.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Heatmap of the 7 x 4 experimental condition means.
    pivot = df.pivot(index="MK_replacement_pct", columns="curing_age_days", values=TARGET)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set(title="Heatmap of Mean Compressive Strength",
           xlabel="Curing age (days)", ylabel="Metakaolin replacement (%)")
    ax.set_xticks(range(len(pivot.columns)), labels=[str(int(v)) for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), labels=[f"{v:g}" for v in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean compressive strength (MPa)")
    fig.tight_layout()
    fig.savefig(FIGURES / "03_strength_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4. Actual versus predicted for the six reserved holdout conditions.
    actual = test[TARGET].to_numpy()
    predicted = np.asarray(predictions)
    low = float(min(actual.min(), predicted.min()))
    high = float(max(actual.max(), predicted.max()))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(actual, predicted)
    ax.plot([low, high], [low, high], linestyle="--", label="Ideal agreement")
    ax.set(title="Actual versus Predicted Compressive Strength (Holdout)",
           xlabel="Actual mean compressive strength (MPa)",
           ylabel="Predicted mean compressive strength (MPa)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "04_holdout_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 5. Residuals against predicted values; residual = prediction - actual.
    residuals = predicted - actual
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(predicted, residuals)
    ax.axhline(0, linestyle="--", label="Zero residual")
    ax.set(title="Holdout Residuals versus Predicted Strength",
           xlabel="Predicted mean compressive strength (MPa)",
           ylabel="Residual (predicted - actual) (MPa)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "05_holdout_residuals_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    df = pd.read_excel(DATA_PATH, sheet_name="Modelling_Dataset")
    required = set(FEATURES + [TARGET, "dataset_split", "group_id"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if len(df) != 28 or df["group_id"].nunique() != 28:
        raise ValueError("Expected 28 unique experimental-condition group means.")
    if set(df["dataset_split"]) != {"train", "test"}:
        raise ValueError("Split labels must be train and test.")
    train = df[df.dataset_split == "train"].copy()
    test = df[df.dataset_split == "test"].copy()
    if len(train) != 22 or len(test) != 6:
        raise ValueError(f"Expected 22 train / 6 test rows; found {len(train)} / {len(test)}.")

    # Training-only repeated CV. Small-fold R² is unstable; interpret errors primarily.
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    cv_results = cross_validate(
        build_model(), train[FEATURES], train[TARGET], cv=cv,
        scoring={"mae": "neg_mean_absolute_error",
                 "rmse": "neg_root_mean_squared_error",
                 "r2": "r2"},
        return_train_score=False,
    )
    cv_summary = {
        "n_splits_total": int(len(cv_results["test_mae"])),
        "note": "Repeated folds are not independent experiments; fold R² is unstable with this small dataset.",
        "mean_MAE_MPa": float(-cv_results["test_mae"].mean()),
        "sd_MAE_MPa": float(cv_results["test_mae"].std(ddof=1)),
        "median_MAE_MPa": float(np.median(-cv_results["test_mae"])),
        "mean_RMSE_MPa": float(-cv_results["test_rmse"].mean()),
        "sd_RMSE_MPa": float(cv_results["test_rmse"].std(ddof=1)),
        "mean_R2": float(cv_results["test_r2"].mean()),
        "sd_R2": float(cv_results["test_r2"].std(ddof=1)),
        "median_R2": float(np.median(cv_results["test_r2"])),
    }

    # Freeze and evaluate the research model on the reserved six points.
    research_model = build_model().fit(train[FEATURES], train[TARGET])
    pred = research_model.predict(test[FEATURES])
    diagnostics = test[["group_id", *FEATURES, TARGET]].copy()
    diagnostics["prediction_MPa"] = pred
    diagnostics["residual_pred_minus_actual_MPa"] = pred - test[TARGET].to_numpy()
    diagnostics["absolute_error_MPa"] = np.abs(pred - test[TARGET].to_numpy())
    diagnostics.to_csv(ARTIFACTS / "research_holdout_predictions.csv", index=False)
    create_figures(df, test, pred)
    research_metrics = {
        "n_test": int(len(test)),
        "MAE_MPa": float(mean_absolute_error(test[TARGET], pred)),
        "RMSE_MPa": float(np.sqrt(mean_squared_error(test[TARGET], pred))),
        "R2": float(r2_score(test[TARGET], pred)),
        "max_absolute_error_MPa": float(np.max(np.abs(pred-test[TARGET].to_numpy()))),
        "interpretation": "Internal six-condition holdout only; small sample from the same experimental programme.",
    }
    joblib.dump(research_model, ARTIFACTS / "research_frozen_22point_model.joblib")

    # Separately fit operational model on all 28 means; no holdout metric is assigned to it.
    operational_model = build_model().fit(df[FEATURES], df[TARGET])
    joblib.dump(operational_model, ARTIFACTS / "operational_model.joblib")
    data_hash = sha256(DATA_PATH)
    metadata = {
        "model_name": "Degree-2 polynomial regression pipeline",
        "fit_data": "All 28 experimental-condition group means (operational refit)",
        "n_fit_observations": int(len(df)),
        "features": FEATURES,
        "target": TARGET,
        "mk_min_pct": float(df[FEATURES[0]].min()),
        "mk_max_pct": float(df[FEATURES[0]].max()),
        "age_min_days": int(df[FEATURES[1]].min()),
        "age_max_days": int(df[FEATURES[1]].max()),
        "degree": 2,
        "polynomial_include_bias": False,
        "preprocessing": "StandardScaler after PolynomialFeatures",
        "regressor": "LinearRegression",
        "dataset_filename": DATA_PATH.name,
        "dataset_sha256": data_hash,
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "warning": "Operational model refitted on all 28 points; frozen 22-point model's holdout scores do not transfer to this artifact.",
    }
    (ARTIFACTS / "operational_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary = {
        "dataset_sha256": data_hash,
        "cross_validation_training_only": cv_summary,
        "frozen_research_model_holdout": research_metrics,
        "validation_separation_note": "Holdout predictions/metrics describe research_frozen_22point_model.joblib. operational_model.joblib was refitted on all 28 observations and has no independent test score.",
    }
    (ARTIFACTS / "research_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("\nArtifacts written to:", ARTIFACTS)
    print("Figures written to:", FIGURES)

if __name__ == "__main__":
    main()
