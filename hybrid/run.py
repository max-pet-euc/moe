"""Run from repository root: python -m hybrid.run"""
from pathlib import Path
import pandas as pd
from .src.load_data import load_settings,load_inputs
from .src.validate_data import validate,prepare
from .src.coverage_report import coverage_summary
from .src.build_features import build_features
from .src.fit_model import fit_markets
from .src.allocate_qs import allocate_periods,aggregate_reporting
from .src.project_outcomes import project_outcomes
from .src.build_report import build_report

def main():
    root=Path(__file__).resolve().parent.parent
    settings=load_settings(root)
    out=root/'hybrid/outputs';out.mkdir(parents=True,exist_ok=True)
    channels,constraints,targets=load_inputs(root,settings)
    channels,targets,excluded=prepare(channels,constraints,targets,settings)
    excluded.to_csv(out/'excluded_channels.csv',index=False)
    coverage_summary(channels,targets,excluded).to_csv(out/'coverage_report.csv',index=False)
    if not excluded.empty:
        print(f'WARNING: {int(excluded["rows"].sum())} channel rows excluded; see hybrid/outputs/excluded_channels.csv')
    errors,warnings,checks=validate(channels,constraints,targets,settings)
    pd.DataFrame([{'level':'error','message':x} for x in errors]+[{'level':'warning','message':x} for x in warnings],columns=['level','message']).to_csv(out/'validation_report.csv',index=False)
    checks.to_csv(out/'floor_checks.csv',index=False)
    for warning in warnings:print('WARNING:',warning)
    if errors:
        for error in errors:print('ERROR:',error)
        raise ValueError(f'{len(errors)} validation error(s); see hybrid/outputs/validation_report.csv')
    features,_=build_features(channels,constraints,targets,settings)
    features.to_csv(out/'features.csv',index=False)
    daily_fit,coefficients=fit_markets(features,settings)
    daily_fit.to_csv(out/'model_diagnostics_daily.csv',index=False)
    coefficients.to_csv(out/'model_coefficients.csv',index=False)
    allocation_daily=allocate_periods(channels,constraints,targets,daily_fit,settings)
    allocation_daily.to_csv(out/'channel_daily_modelled.csv',index=False)
    allocation=aggregate_reporting(allocation_daily,settings)
    projections=project_outcomes(allocation,settings,root)
    projections.to_csv(out/'channel_report.csv',index=False)
    build_report(projections,out/'report.html',settings)
    print('Hybrid run complete:',out/'report.html')

if __name__=='__main__':main()
