# Hybrid configuration

Edit `settings.json` to match the actual QS target file and its columns. Paths are relative to the repository root. `target_filters` can select a target slice (e.g. `{"problem_type":"WEIGHT_LOSS"}`) if your target file has multiple types. The default target column is an assumption, not a verified repository schema.

The optional `cohort_conversion_file` is a CSV with `country_code,brand,problem_type,qs_to_cp,cp_to_op` (or `qs_to_op`), used for projections. Otherwise default conversion assumptions apply and projections are clearly labelled assumptions.

Channel constraint CSV columns: `channel,exposure_metric,minimum_source,minimum_multiplier,constraint_type,enabled,model_group,constraint_period,constraint_weight`. Optional columns are defaulted. `model_group` is retained as metadata in v1; contributions are estimated at individual channel level. Only `minimum_source=last_click_qs` and `constraint_type=hard` or `none` are supported. Monthly is the only supported constraint period.
