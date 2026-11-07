"""Check every paired world against the v7 factorial archive before rerun."""
import hashlib
import json
from pathlib import Path

from src import data
from src.simulate import world_digest

root = Path(__file__).resolve().parents[1]
archive = root / 'results' / 'bpqa_factorial_v7_archive'
inputs = data.load_inputs()
expected = {(dataset, seed, factor) for dataset in ('tdrive', 'brightkite')
            for seed in range(9701, 9709) for factor in (.75, 1.)}
seen = set()
for path in sorted(archive.glob('*.json')):
    record = json.loads(path.read_text(encoding='utf-8'))
    if 'case' not in record:
        continue
    case = record['case']
    key = (case['dataset'], case['seed'], case['budget_factor'])
    if key not in expected or key in seen:
        raise ValueError('Unexpected or duplicate campaign: ' + repr(key))
    if world_digest(data.build_world(case, inputs)) != record['world_sha256']:
        raise ValueError('World differs from v7: ' + repr(key))
    seen.add(key)
if seen != expected:
    raise ValueError('Missing campaigns: ' + repr(sorted(expected - seen)))
for name in ('src/admission_experiments.py', 'src/trust.py', 'src/recovery.py',
             'src/data.py', 'src/admission.py', 'src/_bpqa.py',
             'experiments/run_bpqa_factorial.py', 'experiments/run_coupled_reputation.py'):
    print(hashlib.sha256((root / name).read_bytes()).hexdigest(), name)
print('Verified all 32 world digests against v7.')
