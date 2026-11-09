# MOE Hybrid v1.1 — isolated parallel model

**Does not modify or import the original MOE pipeline.** Only writes under `hybrid/outputs/`.

## Run from MOE repository root

```bash
python -m pip install -r hybrid/requirements.txt
python -m hybrid.run
```

Open `hybrid/outputs/report.html`.

## Real source inputs (already in your repository)

- `data/raw/data_channel.csv`: `country_code,brand,date_day,channel,spend_local,impressions,attributed_qs,minimum_qs_source,minimum_qs`
- `data/raw/data_constraints.csv`: `channel,enabled,model_group,exposure_metric,minimum_source,minimum_multiplier,constraint_type,constraint_period,constraint_weight`
- `data/raw/data_funnel.csv`: `brand,country_code,date_day,qs,cp,op`

No `problem_type` dimension is present. `qs` from `data_funnel.csv` is the actual target. `spend_local` is local currency and is **not** converted to GBP. `minimum_qs` is the last-click/business minimum and is not the platform conversion signal. `attributed_qs` can exceed actual QS in aggregate.

The loader interprets `-` as missing, not zero. Configuration uses `minimum_source=minimum_qs` and `constraint_period=month` directly. `minimum_qs_source` is retained as source metadata; its values (`att`, `hdyhau`) do not change the floor rule.

## Output files

- `report.html`: separate country/brand tabs and monthly comparisons of attributed vs modelled QS and CPQS.
- `channel_monthly.csv`: allocation and projected outcomes.
- `excluded_channels.csv`: every unmapped/disabled channel and market lacking a funnel target, including excluded spend. **Review this before trusting totals.**
- `validation_report.csv` and `floor_checks.csv`: validation diagnostics.
- `features.csv`, `model_diagnostics_daily.csv`, `model_coefficients.csv`: model diagnostics.

## Important constraints and limitations

- Only configured and enabled channels with a matching funnel market are modelled. Unknown channels (e.g. `gifting`, `production`, `big_influencer`, blank channels) are **excluded and audited**, not assigned automatically. Likewise, channel markets with no matching funnel actuals are excluded and audited.
- Monthly hard floors must sum to no more than monthly actual QS. Missing hard floor values are errors. The daily model uses both impressions (or spend, per configuration) and attributed QS as signals.
- **The output is a historical constrained allocation, not causal incrementality.** Same-day attributed QS is downstream of advertising and may create target leakage for forecasting.
- `projected_op` and `projected_cohorted_cac` still use configured *placeholder* funnel conversion rates unless `cohort_conversion_file` is provided. The current funnel contains uncohorted daily CP and OP, not cohort-specific rates; the model does not silently call these cohorted rates.
- `model_group` is currently not used to combine channels; it is retained for a future model-grouping implementation.
- Spend and CPQS/CAC are in **local market currency**, not a common currency.
- Markets need at least `minimum_training_days` of funnel target data; adjust `hybrid/config/settings.json` only if appropriate.

The original model can continue to be run independently. Its validator may need a separate compatibility adjustment if it scans all files in `data/raw/`; this package does not edit it.
