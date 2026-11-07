import argparse,json,os,subprocess,sys,time
from pathlib import Path
from transport import Peer

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--source',required=True);a=p.parse_args()
    code=Path(__file__).resolve().parent;root=Path(a.root);logs=root/'logs';logs.mkdir(exist_ok=True)
    procs={}
    for role in ('ledger','requester','broker'):
        log=(logs/(role+'.log')).open('w')
        proc=subprocess.Popen([sys.executable,str(code/'services.py'),role,'--root',str(root),'--source',a.source],
             stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=os.environ.copy())
        procs[role]=proc;log.close()
    (root/'pids.json').write_text(json.dumps({r:p.pid for r,p in procs.items()}))
    peer=Peer(root,'controller')
    for role in procs:
        for _ in range(60):
            if procs[role].poll() is not None:raise RuntimeError(role+' failed: '+(logs/(role+'.log')).read_text()[-3000:])
            try:
                print(peer.call(role,'health',{}),flush=True);break
            except OSError:time.sleep(.2)
        else:raise RuntimeError('No readiness response '+role)

if __name__=='__main__':main()
