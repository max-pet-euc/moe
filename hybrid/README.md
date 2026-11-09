# MOE Hybrid v1.2

This package is independent of the original MOE pipeline. It reads existing
`data/raw/data_channel.csv`, `data/raw/data_constraints.csv`, and
`data/raw/data_funnel.csv` without modifying them.

## Run

From the repository root:

```bash
python -m pip install -r hybrid/requirements.txt
python -m hybrid.run
```

## Settings

In `hybrid/config/settings.json`:

- `reporting.frequency`: `daily`, `weekly`, or `monthly`.
- `constraints.frequency`: `daily`, `weekly`, or `monthly`.
- Weekly periods begin Monday; monthly periods begin the first of the month.
- Missing funnel actuals for channel days are excluded and audited, not replaced with zero.
- Missing hard minimums are **errors**, not zeros.
- Hard minimums are validated at the selected constraint frequency.

The `constraint_period` field in the raw constraints file is not used to
select frequency in v1.2; the global settings frequency applies to all
channels. This avoids conflicting period definitions.

## Outputs

All written under `hybrid/outputs/`:

- `coverage_report.csv`: included/excluded row and spend coverage by market.
- `excluded_channels.csv`: unmapped and daily target-missing exclusions.
- `validation_report.csv` and `floor_checks.csv`: checks and hard-floor feasibility.
- `features.csv`, `model_diagnostics_daily.csv`, `model_coefficients.csv`.
- `channel_daily_modelled.csv`: daily constrained allocations.
- `channel_report.csv`: aggregated to reporting frequency with CPQS and projections.
- `report.html`: interactive market tabs and reporting periods.

## Interpretation

The model uses exposure and platform-attributed QS as two explanatory signals.
It produces a **constrained historical QS contribution allocation**, not a
causally identified estimate of incremental QS. Last-click floors are business
constraints. Projected OP/CAC use placeholder conversion assumptions until
actual cohort rates are connected. Cost values are in local currency.

Reporting frequency and constraint frequency can differ. A weekly report
under monthly constraints is not guaranteed to meet weekly floors.


### v1.3: insufficient training history
Markets with fewer than `minimum_training_days` target days are skipped rather than stopping the entire run.
`outputs/market_status.csv` lists valid and invalid markets and the number of available days.
The HTML report includes an **Invalid data** tab for each skipped market.
Other markets continue to fit and allocate normally. Global validation errors still stop the run.
