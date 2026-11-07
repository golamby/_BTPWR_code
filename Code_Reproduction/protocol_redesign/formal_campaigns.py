"""Serial, hash-bound full protocol campaigns; offline replay excludes auditor overhead."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args()
base=Path('/root/autodl-tmp/codex-btpwr-review2');code=Path(__file__).resolve().parent
root=base/'e2e'/('q36_reference_check_hardened' if a.check_only else 'formal_q36_hardened');root.mkdir(exist_ok=True)
source=root/'source'
if not source.exists():
    source.mkdir();shutil.copytree(base/'run_traditional/src',source/'src',ignore=shutil.ignore_patterns('__pycache__'))
    (source/'data').symlink_to(base/'run_traditional/data',target_is_directory=True)
env=dict(os.environ,PYTHONPATH=':'.join(str(base/x) for x in ['e2e/chain-deps','e2e/deps','deps'])+':.',OPENBLAS_NUM_THREADS='1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(root/'source_manifest.json').write_text(json.dumps({str(f.relative_to(source)):sha(f) for f in (source/'src').glob('*.py')},indent=2))
cases=[(80,'tdrive')] if a.check_only else [(n,d) for n in [80,100] for d in ['tdrive','foursquare_tky','brightkite']]
state=[]
for n,d in cases:
    case=root/(d+'_n'+str(n));case.mkdir(exist_ok=True);fixture=case/'private_fixture';out=case/'results';receipt=case/'complete.json'
    if receipt.exists():state.append(json.loads(receipt.read_text()));continue
    if fixture.exists():raise ValueError('Incomplete prior fixture requires investigation '+str(fixture))
    log=(case/'campaign.log').open('w');start=time.time();stages={}
    def run(script,args,timeout=1800):
        t=time.perf_counter();subprocess.run([sys.executable,str(code/script)]+args,env=env,cwd=str(code),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout);stages[script]=time.perf_counter()-t
    record=dict(dataset=d,workers=n,rounds=2 if a.check_only else 50,status='running',started_epoch=start)
    (root/'state.json').write_text(json.dumps(dict(active=record,completed=state),indent=2))
    try:
        run('prepare.py',['--source',str(source),'--output',str(fixture),'--workers',str(n),'--dataset',d,'--seed','9801','--budget','450'])
        run('launch_roles.py',['--root',str(fixture),'--source',str(source)])
        run('run_clients.py',['--root',str(fixture),'--source',str(source),'--rounds',str(record['rounds']),'--output',str(out)])
        checks=json.loads((out/'reference_checks.json').read_text());timing=json.loads((out/'timing.json').read_text())
        assert len(checks)==record['rounds']==timing['completed_rounds']
        record.update(status='completed',wall_seconds=time.time()-start,stages=stages,input_world_sha256=sha(fixture/'controller/world.npz'),config_sha256=sha(fixture/'public.json'),
          protocol_source_sha256={f.name:sha(f) for f in code.glob('*.py')},outputs={f.name:sha(f) for f in out.iterdir() if f.is_file()})
        receipt.write_text(json.dumps(record,indent=2));state.append(record)
        print(json.dumps({k:record[k] for k in ['dataset','workers','rounds','status','wall_seconds']}),flush=True)
    except Exception as e:
        record.update(status='failed',error=repr(e));(case/'failure.json').write_text(json.dumps(record,indent=2));raise
    finally:
        if (fixture/'pids.json').exists():run('stop_roles.py',['--root',str(fixture)])
        log.close();(root/'state.json').write_text(json.dumps(dict(active=None,completed=state),indent=2))
print('ALL_COMPLETE',flush=True)
