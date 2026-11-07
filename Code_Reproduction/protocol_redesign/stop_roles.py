import argparse,json,os,signal
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    root=Path(a.root).resolve()
    for role,pid in json.loads((root/'pids.json').read_text()).items():
        proc=Path('/proc')/str(pid)/'cmdline'
        if not proc.exists():continue
        args=proc.read_bytes().replace(b'\0',b' ').decode()
        if 'services.py' not in args or str(root) not in args:raise ValueError('PID ownership mismatch')
        os.kill(pid,signal.SIGTERM);print('stopped',role,pid)

if __name__=='__main__':main()
