"""Seal PEC210 exploratory source, selected inputs and launch recipe."""
from common import *
import subprocess

p=load_json(OUT/'manifest/protocol.json')
rows=jsonl(OUT/'manifest/runs.jsonl')
checks=load_json(OUT/'validation/deterministic_checks.json')
assert len(rows)==10 and checks['status']=='PASS' and checks['new_measurement_samples']==0
ph=digest(OUT/'manifest/protocol.json')
launch=OUT/'launch_pec210_explore.sbatch'
with launch.open('x') as stream:
    stream.write(f'''#!/bin/bash
#SBATCH --job-name=v5_pec210_explore
#SBATCH --partition=cpu-skylake
#SBATCH --array=0-9%2
#SBATCH --dependency=afterany:5306944
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=96:00:00
#SBATCH --output={OUT}/logs/pec210_%A_%a.out
#SBATCH --error={OUT}/logs/pec210_%A_%a.err
set -euo pipefail
cd {ROOT}
source {OUT}/source/environment.sh {OUT}
{ROOT}/.venv/bin/python -B {OUT}/source/launch_preflight.py
srun --ntasks=1 --cpu-bind=threads {ROOT}/.venv/bin/python -B {OUT}/source/runner.py --index "$SLURM_ARRAY_TASK_ID" --protocol-sha256 {ph}
''')
subprocess.run(['bash','-n',str(launch)],check=True)
lineage=load_json(OUT/'manifest/baseline_lineage.json')
paths=set((OUT/'source').glob('*.py'))|set((OUT/'source').glob('*.sh'))
paths|=set(FROZEN.rglob('*.py'))
paths|=set((OUT/'manifest').glob('*.json'))|{OUT/'manifest/runs.jsonl',launch}
paths|=set((OUT/'validation').glob('*.json'))
paths|={Path(lineage[k]) for k in ['base_manifest','corrected_reference_manifest',
                                    'p01_p02_manifest','corrected_teacher_snapshot']}
paths.add(ROOT/'artifacts/strict_v5_patience100_v1_20261019T140521Z/source/pec_kernel.py')
for row in rows:
    for key in ['teacher_path','train_split','validation_split','initial_path','orders_path',
                'margin_target_registry','classification_target_registry']:
        paths.add(Path(row[key]))
assert all('/datasets/test/' not in str(path) and 'test_prediction' not in path.name for path in paths)
new_json(OUT/'manifest/frozen_execution.json',dict(utc=now(),protocol_sha256=ph,
    files=[source_record(path) for path in sorted(paths)],
    policy='Immutable source, targets and input arrays after submission; acquired costs/results appended separately.'))
print(json.dumps(dict(status='FROZEN',protocol_sha256=ph,
    frozen_sha256=digest(OUT/'manifest/frozen_execution.json'),files=len(paths))))
