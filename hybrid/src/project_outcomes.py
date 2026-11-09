import pandas as pd

GROUP=['country_code','brand','problem_type']

def project_outcomes(allocation, settings, root):
    if allocation.empty: return allocation
    out=allocation.copy()
    out['qs_to_cp']=float(settings['default_qs_to_cp'])
    out['cp_to_op']=float(settings['default_cp_to_op'])
    out['qs_to_op']=settings.get('default_qs_to_op')
    path=settings.get('cohort_conversion_file')
    if path:
        assumptions=pd.read_csv(root/path)
        for col in GROUP:
            assumptions[col]=assumptions[col].astype(str).str.lower()
        out=out.merge(assumptions,on=GROUP,how='left',suffixes=('','_override'))
        for col in ['qs_to_cp','cp_to_op','qs_to_op']:
            alt=f'{col}_override'
            if alt in out:
                out[col]=out[alt].combine_first(out[col]); out=out.drop(columns=alt)
    out['projected_cp']=out['modelled_qs']*out['qs_to_cp']
    out['projected_op']=out['modelled_qs']*out['qs_to_op'].where(out['qs_to_op'].notna(),out['qs_to_cp']*out['cp_to_op'])
    out['projected_cohorted_cac']=out['spend']/out['projected_op'].replace(0,float('nan'))
    return out
