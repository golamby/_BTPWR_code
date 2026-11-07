"""Validate and summarize nine serial numerical timing runs."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]

def main():
    folder=ROOT/'timing'/'results'; reference=ROOT/'final_results'/'review_sweeps';rows=[]
    for path in sorted((folder/'serial_timing').glob('sweep*.json')):
        j=json.loads(path.read_text());r=json.loads((reference/path.name).read_text())
        bp=json.loads((folder/'bpqa_native_sweeps'/path.name).read_text())
        if j['world_sha256']!=r['world_sha256'] or bp['world_sha256']!=r['world_sha256']:
            raise ValueError('Timing world mismatch: '+path.name)
        for item in j['results']:
            rows.append(dict(dataset=j['case']['dataset'],seed=j['case']['seed'],method=item['method'],seconds=item['seconds']))
        rows.append(dict(dataset=j['case']['dataset'],seed=j['case']['seed'],method='bpqa',seconds=bp['seconds']))
    df=pd.DataFrame(rows)
    if len(df)!=72:raise ValueError('Expected nine worlds and eight methods')
    df.to_csv(ROOT/'final_results'/'tables'/'serial_timing_cases.csv',index=False)
    summary=df.groupby('method').seconds.agg(['mean','min','max'])
    summary.to_csv(ROOT/'final_results'/'tables'/'serial_timing_summary.csv')
    print(summary.round(3).to_string())

if __name__=='__main__':main()
