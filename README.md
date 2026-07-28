# MOE pipeline v1

Copy the files into the matching repository locations.

The notebook:

1. validates every CSV in `data/raw/`
2. writes `data/outputs/validation_report.csv`
3. stops on validation failures
4. reshapes and joins the raw datasets
5. creates basic lag, rolling and log features
6. writes `data/engineered/data_features.csv`

Expected structures:

data_media_inputs.csv
grain: one row per date_day
format: wide spend columns

data_digital_platforms.csv
grain: date_day × platform × channel × metric
columns: date_day, platform, channel, metric, value

data_attribution.csv
grain: date_day × platform × channel × metric
columns: date_day, platform, channel, metric, value

data_external.csv
grain: one row per date_day
format: wide external metrics




