# MOE Hybrid v1 (independent prototype)

This is an additive, non-destructive prototype. It does not import or edit the original `src/` modules and writes exclusively to `hybrid/outputs/`.

## Installation and run

From the **MOE repository root**:

```bash
python -m pip install -r hybrid/requirements.txt
python -m hybrid.run
```

Open `hybrid/outputs/report.html` in a browser.

## Inputs

Configure file names in `hybrid/config/settings.json`. Defaults assume:

- `data/raw/channel_daily.csv`: `date_day,country_code,brand,problem_type,channel,spend,impressions,attributed_qs,last_click_qs`
- `data/raw/channel_constraints.csv`: `channel,exposure_metric,minimum_source,minimum_multiplier,constraint_type,enabled,model_group,constraint_period,constraint_weight`
- `data/raw/data_targets.csv`: `date_day,country_code,brand,problem_type,qs` **(target schema must be verified)**

If your new raw files use different names, update settings rather than renaming existing data. `impressions` and `attributed_qs` are retained as separate model features; the configured exposure source selects impressions or spend for each channel. Both may exceed or be absent independently. Missing signal values are flagged as missing and currently imputed to zero for model fitting; this should be refined once actual data coverage is inspected.

## What it does

1. Reads new channel data and existing daily actual QS targets.
2. Validates uniqueness, nonnegative values, channel mapping, and **monthly** last-click floor <= actual QS for each market.
3. Fits independent per-market positive Ridge models using log impressions/spend and log attributed QS, plus basic time controls.
4. Calculates nonnegative channel evidence from the fitted coefficients.
5. Solves a constrained monthly allocation that reconciles channel QS + baseline to actual QS and enforces last-click floors.
6. Produces platform CPQS, modelled CPQS, projected CP/OP/CAC, and separate HTML tabs per market.

## Critical interpretation

- **The fitted model is associative, not causal.** Contemporaneous attributed QS is partly an outcome and creates leakage for forward-looking budget scenarios. This version must **not** be used to claim incrementality or optimise spend.
- **The monthly allocation is a reconciliation step**, not the raw Ridge prediction. Last-click is a configurable *business floor*, not a causal floor.
- `model_group` is preserved but **not yet implemented as an aggregation**. Channel models are currently individual.
- Daily channel-level contributions are not reconciled. Monthly channel contributions are.
- `constraint_weight` is reserved for future soft constraints; v1 supports hard or none.
- Conversion assumptions are placeholders until actual cohort rates are supplied. Projected CAC is spend / projected OP; it is not observed CAC.
- The code intentionally has no write operations outside `hybrid/outputs/`.

## Next steps before production

Verify raw CSV schemas and target grain, compare with the original model, add train/test backtesting, tune baseline allocation, model lags and saturation, and build a distinct causal response model for incremental QS. Do not treat the current report as a validated marketing effectiveness estimate.
