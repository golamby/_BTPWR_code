"""One-round integration check only; never launches formal campaigns."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

code = Path(__file__).resolve().parent
base = Path('/root/autodl-tmp/codex-btpwr-review2')
fixture = base / 'e2e/default80_functional_fixture'
output = base / 'e2e/default80_functional_check'
source = base / 'run_traditional'
env = dict(os.environ)
env['PYTHONPATH'] = ':'.join(str(base / p) for p in ['e2e/chain-deps', 'e2e/deps', 'deps']) + ':.'
env['OPENBLAS_NUM_THREADS'] = '1'

def run(name, *args):
    subprocess.run([sys.executable, str(code / name)] + list(args), env=env, cwd=str(code), check=True)

if fixture.exists() or output.exists():
    raise ValueError('Refuse to overwrite an existing functional check')
run('prepare.py', '--source', str(source), '--output', str(fixture), '--workers', '80', '--budget', '450')
try:
    run('launch_roles.py', '--root', str(fixture), '--source', str(source))
    run('run_clients.py', '--root', str(fixture), '--source', str(source), '--rounds', '1', '--output', str(output))
    timing = json.loads((output / 'timing.json').read_text())
    assert timing['workers'] == 80 and timing['completed_rounds'] == 1
    receipt = dict(kind='functional_only_not_formal_performance_evidence', workers=80, rounds=1,
                   completed=True, source_sha256={name:hashlib.sha256((code/name).read_bytes()).hexdigest()
                   for name in ['prepare.py','services.py','run_clients.py']})
    (output / 'check.json').write_text(json.dumps(receipt, indent=2))
finally:
    if (fixture / 'pids.json').exists():
        run('stop_roles.py', '--root', str(fixture))
