# Marketing Optimisation Engine

MOE is a configuration-driven framework for building daily marketing features,
training forecasting models, and running budget optimisation scenarios.

## Repository structure

```text
config/             Project, target, channel, and feature settings
data/raw/           Source data only
data/engineered/    Generated feature datasets
data/modelling/     Saved model artefacts
data/outputs/       Reports and optimisation outputs
src/                Reusable Python modules
notebooks/          Pipeline entry points
```

## Pipeline

1. Add source files to `data/raw/`
2. Run `notebooks/01_feature_builder.ipynb`
3. Create `data/engineered/data_features.csv`
4. Run `notebooks/02_train_model.ipynb`
5. Run optimisation and reporting notebooks

## Raw datasets

data_funnel_uncohorted.csv
    Daily event counts based on event date.

data_funnel_cohorted.csv
    Daily funnel counts attributed to the cohort entry date (QS date).

## Status

Initial project skeleton only.
