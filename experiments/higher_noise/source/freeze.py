"""Seal source, targets, teacher identities and scheduler inputs before launch."""
from common import *
rows=jsonl(OUT/'manifest/runs.jsonl')
paths=set((OUT/'source').glob('*.py'))|set((OUT/'frozen_src').rglob('*.py'))
paths|=set(OUT.glob('launch_*.sbatch'))
paths|={OUT/'source/environment.sh',OUT/'manifest/protocol.json',
        OUT/'manifest/runs.jsonl',OUT/'manifest/scheduler_groups.json'}
paths|=set((OUT/'manifest').glob('*_indices.txt'))
paths|={Path(r['teacher_path']) for r in rows}
paths|={Path(r['margin_target_registry']) for r in rows}
paths|={Path(r['classification_target_registry']) for r in rows}
payload=dict(schema='strict_v5.lowshot_depth_noise_3x3_dual.frozen_execution.v1',
             sealed_at_utc=now(),
             source_campaigns=['strict_v5_patience100_v1_20261019T140521Z',
                               'strict_v5_pec180_boost25_classification_v1_20261019T092910Z'],
             files=[source_record(p) for p in sorted(paths)])
new_json(OUT/'manifest/frozen_execution.json',payload)
print(dict(files=len(paths),frozen_sha256=digest(OUT/'manifest/frozen_execution.json')))
