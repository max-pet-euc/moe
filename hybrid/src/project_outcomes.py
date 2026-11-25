import pandas as pd
import numpy as np

GROUP=['country_code','brand']

def project_outcomes(allocation, settings, root, actuals=None):
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
    mode=settings.get('conversion_rate_source','projected_cohort')
    if mode not in ('projected_cohort','fixed'):
        raise ValueError('conversion_rate_source must be projected_cohort or fixed')
    out['qs_to_op']=out['qs_to_op'].where(out['qs_to_op'].notna(),out['qs_to_cp']*out['cp_to_op'])
    out['conversion_rate_used']='fixed' if mode=='fixed' else 'fallback assumptions'
    if mode=='projected_cohort' and actuals is not None:
        keys=GROUP+['period_start']
        cols=keys+['actual_qs','actual_op_projection','actual_cp_projection']
        benchmark=actuals[cols].rename(columns={'actual_qs':'cohort_qs'})
        out=out.merge(benchmark,on=keys,how='left',validate='many_to_one')
        q=out['cohort_qs']; op=out['actual_op_projection']; cp=out['actual_cp_projection']
        valid=np.isfinite(q)&np.isfinite(op)&(q>0)&(op>=0)&(op<=q)
        split=valid&np.isfinite(cp)&(cp>=op)&(cp<=q)&(cp>0)
        out.loc[valid,'qs_to_op']=op[valid]/q[valid]
        # Without cohort CP, only the combined rate is observed. Do not infer a split.
        out.loc[valid,['qs_to_cp','cp_to_op']]=np.nan
        out.loc[split,'qs_to_cp']=cp[split]/q[split]
        out.loc[split,'cp_to_op']=op[split]/cp[split]
        out.loc[valid,'conversion_rate_used']='projected cohort (QS to OP)'
        out.loc[split,'conversion_rate_used']='projected cohort (QS to CP to OP)'
        out=out.drop(columns=['cohort_qs','actual_op_projection','actual_cp_projection'])
    out['projected_cp']=out['modelled_qs']*out['qs_to_cp']
    out['projected_op']=out['modelled_qs']*out['qs_to_op'].where(out['qs_to_op'].notna(),out['qs_to_cp']*out['cp_to_op'])
    out['projected_cohorted_cac']=out['spend']/out['projected_op'].replace(0,float('nan'))
    return out
