# MK Concrete Strength Predictor

A Streamlit-based research demonstration for estimating mean cube compressive strength (MPa) from metakaolin (MK) replacement percentage and curing age.

## Dataset and scope
- Input: `Concrete_Modelling_Dataset_Step4A.xlsx`
- Expected: 28 unique MK–age group means; 22 `train`, 6 `test`.
- Predictors: `MK_replacement_pct`, `curing_age_days`
- Target: `mean_MPa`
- Model: `PolynomialFeatures(degree=2, include_bias=False)` -> `StandardScaler` -> `LinearRegression`

The app limits user inputs to the observed envelope (0–30% MK and 7–90 days) and offers the experimental ages 7, 28, 56 and 90 days. Staying within the numerical envelope does not mean every combination is directly experimentally observed.

## Important model distinction
`train_model.py` creates two artifacts:
1. `research_frozen_22point_model.joblib`: fit on the 22 declared training rows and evaluated on the six reserved rows. Its diagnostics and scores are research validation outputs.
2. `operational_model.joblib`: separately refit on all 28 group means for application use. **The research holdout scores do not apply to this refitted model.**

The training script also reruns repeated 5-fold CV on training data only. Results may vary slightly with library versions. Do not repeatedly tune to the six test observations.

## Windows setup (PowerShell)

### Requirements
- Python installed on your system.
- The repository cloned or downloaded locally.

### Install dependencies

Open PowerShell in the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run the application

The application uses the pre-trained operational model in:

`artifacts/operational_model.joblib`

Run:

```powershell
streamlit run app.py
```

The application requires the operational model and its metadata to be present in the `artifacts/` directory.

### Dataset availability and model retraining

The research dataset is not included in this repository because permission to redistribute it has not yet been confirmed.

The training script requires:

`Concrete_Modelling_Dataset_Step4A.xlsx`

To retrain or reproduce the research workflow, obtain an authorised copy of the dataset and place it in the project root under the expected filename.

Then run:

```powershell
python train_model.py
```

Do not interpret successful application launch as independent validation of the model.

## Outputs

The repository includes the pre-trained operational model required by the application:

- `operational_model.joblib`
- `operational_model_metadata.json`

The operational model is refitted on all 28 experimental condition means for application use.

The frozen research model, holdout predictions, validation summary, and original experimental dataset are not included in this deployment repository.

Retraining requires an authorised copy of the original research dataset, as described above.