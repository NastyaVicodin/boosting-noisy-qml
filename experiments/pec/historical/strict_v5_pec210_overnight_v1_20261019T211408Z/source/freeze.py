from common import *
import importlib, importlib.metadata, subprocess

p=load_json(OUT/'manifest/protocol.json');base=Path(p['source_campaign'])
versions={name:importlib.metadata.version(name) for name in ['jax','jaxlib','equinox','optax','numpy','tensorcircuit']}
modules={}
for name in ['boosting_qml.pqc.density_matrix','boosting_qml.pqc.measurement','boosting_qml.pec.explicit_gatewise','boosting_qml.training.convergence','boosting_qml.weak_learners.pqc_binary','numpy_compat']:
    module=importlib.import_module(name);assert Path(module.__file__).resolve().is_relative_to(FROZEN.resolve());modules[name]=source_record(module.__file__)
new_json(OUT/'manifest/runtime.json',dict(utc=now(),versions=versions,modules=modules,executable=sys.executable))
ph=digest(OUT/'manifest/protocol.json')
for label,array,hours,memory,entry in [
    ('pec210','#SBATCH --array=0-9%10\n',96,'8G',f'{ROOT}/.venv/bin/python -B {OUT}/source/launch_preflight.py\nsrun --ntasks=1 --cpu-bind=threads {ROOT}/.venv/bin/python -B {OUT}/source/runner.py --index "$SLURM_ARRAY_TASK_ID" --protocol-sha256 {ph}'),
    ('report','',1,'1G',f'{ROOT}/.venv/bin/python -B {OUT}/source/report.py')]:
    script=f'''#!/bin/bash
#SBATCH --job-name=v5_overnight_{label}
#SBATCH --partition=cpu-skylake
{array}#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={memory}
#SBATCH --time={hours}:00:00
#SBATCH --output={OUT}/logs/{label}_%A_%a.out
#SBATCH --error={OUT}/logs/{label}_%A_%a.err
set -euo pipefail
cd {ROOT}
source {OUT}/source/environment.sh {OUT}
{entry}
'''
    with (OUT/f'launch_{label}.sbatch').open('x') as f:f.write(script)
    subprocess.run(['bash','-n',str(OUT/f'launch_{label}.sbatch')],check=True)
paths=set((OUT/'source').glob('*.py'))|set((OUT/'source').glob('*.sh'))|set(FROZEN.rglob('*.py'))|set((OUT/'manifest').glob('*.json'))|{OUT/'manifest/runs.jsonl'}|set(OUT.glob('*.sbatch'))|set((OUT/'validation').glob('*.json'))
paths.add(base/'source/report.py');paths.add(base/'source/common.py');paths.add(base/'source/patience.py')
for row in jsonl(OUT/'manifest/runs.jsonl'):
    for key in ['train_split','validation_split','initial_path','orders_path','margin_target_registry','target_registry','teacher_path']:paths.add(Path(row[key]))
assert all('/datasets/test/' not in str(p) and 'test_prediction' not in p.name for p in paths)
new_json(OUT/'manifest/frozen_execution.json',dict(utc=now(),protocol_sha256=ph,files=[source_record(p) for p in sorted(paths)],policy='Immutable after submission; acquired costs and reports are appended separately.'))
print(json.dumps(dict(protocol_sha256=ph,frozen_sha256=digest(OUT/'manifest/frozen_execution.json'),files=len(paths))))
