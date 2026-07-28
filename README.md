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

- Media inputs: `date_day,channel,spend,impressions,clicks`
- Digital platforms: `date_day,metric,value`
- External: `date_day,metric,value`
- Attribution: either channel-wide or metric/value format
