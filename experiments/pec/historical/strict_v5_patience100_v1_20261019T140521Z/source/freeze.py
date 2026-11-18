from common import *
import importlib,importlib.metadata,subprocess

versions={name:importlib.metadata.version(name) for name in ['jax','jaxlib','equinox','optax','numpy','tensorcircuit']}
modules={}
for name in ['boosting_qml.pqc.density_matrix','boosting_qml.pqc.measurement','boosting_qml.pec.explicit_gatewise','boosting_qml.training.convergence','boosting_qml.weak_learners.pqc_binary','numpy_compat']:
    module=importlib.import_module(name);assert Path(module.__file__).resolve().is_relative_to(FROZEN.resolve());modules[name]=source_record(module.__file__)
new_json(OUT/'manifest/runtime.json',dict(utc=now(),versions=versions,modules=modules,executable=sys.executable))
ph=digest(OUT/'manifest/protocol.json')
for arm,indices,hours in [('pec120','0-5%2',48),('pec180','6-13%2',96),('reference25','14-23%2',4),('review120','24-27%2',48),('review180',None,8),('guard',None,96),('report',None,1)]:
    array='' if indices is None else f'#SBATCH --array={indices}\n'
    memory='1G' if arm in ['guard','report'] else '8G'
    entry=(f'{ROOT}/.venv/bin/python -B {OUT}/source/guard.py --index "$1"' if arm=='guard' else
           f'{ROOT}/.venv/bin/python -B {OUT}/source/launch_preflight.py\nsrun --ntasks=1 --cpu-bind=threads {ROOT}/.venv/bin/python -B {OUT}/source/runner.py --index "$SLURM_ARRAY_TASK_ID" --protocol-sha256 {ph}')
    if arm=='report':entry=f'{ROOT}/.venv/bin/python -B {OUT}/source/report.py --final'
    script=f'''#!/bin/bash
#SBATCH --job-name=v5_p100_{arm}
#SBATCH --partition=cpu-skylake
{array}#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem={memory}
#SBATCH --time={hours}:00:00
#SBATCH --output={OUT}/logs/{arm}_%A_%a.out
#SBATCH --error={OUT}/logs/{arm}_%A_%a.err
set -euo pipefail
cd {ROOT}
source {OUT}/source/environment.sh {OUT}
{entry}
'''
    with (OUT/f'launch_{arm}.sbatch').open('x') as f:f.write(script)
    subprocess.run(['bash','-n',str(OUT/f'launch_{arm}.sbatch')],check=True)
paths=set((OUT/'source').glob('*.py'))|set((OUT/'source').glob('*.sh'))|set(FROZEN.rglob('*.py'))|set((OUT/'manifest').glob('*.json'))|{OUT/'manifest/runs.jsonl'}|set(OUT.glob('*.sbatch'))|set((OUT/'validation').glob('*.json'))|{OUT/'LAUNCH_PLAN.md',OUT/'STOPPING_POLICY.md'}
for row in jsonl(OUT/'manifest/runs.jsonl'):
    for key in ['train_split','validation_split','initial_path','orders_path','margin_target_registry','target_registry','teacher_path']:paths.add(Path(row[key]))
assert all('/datasets/test/' not in str(p) and 'test_prediction' not in p.name for p in paths)
new_json(OUT/'manifest/frozen_execution.json',dict(utc=now(),protocol_sha256=ph,files=[source_record(p) for p in sorted(paths)],policy='Immutable after submission; runtime outputs and source checkpoint inventories are appended separately.'))
print(json.dumps(dict(protocol_sha256=ph,frozen_sha256=digest(OUT/'manifest/frozen_execution.json'),files=len(paths))))
