"""Seal the prepared PEC240 sources, inputs and launch script; no sampling."""
from common import *
import importlib, importlib.metadata, subprocess

protocol = load_json(OUT/'manifest/protocol.json')
base = Path(protocol['parent_campaign'])
versions = {name: importlib.metadata.version(name) for name in ['jax','jaxlib','equinox','optax','numpy','tensorcircuit']}
modules = {}
for name in ['boosting_qml.pqc.density_matrix','boosting_qml.pqc.measurement',
             'boosting_qml.pec.explicit_gatewise','boosting_qml.training.convergence',
             'boosting_qml.weak_learners.pqc_binary','numpy_compat']:
    module = importlib.import_module(name)
    assert Path(module.__file__).resolve().is_relative_to(FROZEN.resolve())
    modules[name] = source_record(module.__file__)
new_json(OUT/'manifest/runtime.json',dict(utc=now(),versions=versions,modules=modules,executable=sys.executable))

ph = digest(OUT/'manifest/protocol.json')
launch = OUT/'launch_pec240.sbatch'
with launch.open('x') as stream:
    stream.write(f'''#!/bin/bash
#SBATCH --job-name=v5_pec240_d3
#SBATCH --partition=cpu-skylake
#SBATCH --array=0-9%2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=96:00:00
#SBATCH --output={OUT}/logs/pec240_%A_%a.out
#SBATCH --error={OUT}/logs/pec240_%A_%a.err
set -euo pipefail
cd {ROOT}
source {OUT}/source/environment.sh {OUT}
{ROOT}/.venv/bin/python -B {OUT}/source/launch_preflight.py
srun --ntasks=1 --cpu-bind=threads {ROOT}/.venv/bin/python -B {OUT}/source/runner.py --index "$SLURM_ARRAY_TASK_ID" --protocol-sha256 {ph}
''')
subprocess.run(['bash','-n',str(launch)],check=True)

paths = set((OUT/'source').glob('*.py')) | set((OUT/'source').glob('*.sh'))
paths |= set(FROZEN.rglob('*.py'))
paths |= set((OUT/'manifest').glob('*.json')) | {OUT/'manifest/runs.jsonl',launch}
paths |= set((OUT/'validation').glob('*.json'))
paths |= {base/'manifest/protocol.json',base/'manifest/runs.jsonl',base/'source/pec_kernel.py'}
for row in jsonl(OUT/'manifest/runs.jsonl'):
    for key in ['train_split','validation_split','initial_path','orders_path',
                'margin_target_registry','target_registry','teacher_path']:
        paths.add(Path(row[key]))
assert all('/datasets/test/' not in str(path) and 'test_prediction' not in path.name for path in paths)
new_json(OUT/'manifest/frozen_execution.json',dict(utc=now(),protocol_sha256=ph,
    files=[source_record(path) for path in sorted(paths)],
    policy='Immutable after submission; acquired costs and reports appended separately.'))
print(json.dumps(dict(protocol_sha256=ph,frozen_sha256=digest(OUT/'manifest/frozen_execution.json'),files=len(paths))))
