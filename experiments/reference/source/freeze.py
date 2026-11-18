from common import *
rows=jsonl(OUT/'manifest/runs.jsonl')
paths=set((OUT/'source').glob('*.py'))|set((OUT/'frozen_src').rglob('*.py'))
paths|={OUT/'launch_reference.sbatch',OUT/'source/environment.sh',
        OUT/'manifest/protocol.json',OUT/'manifest/runs.jsonl'}
paths|={Path(r['teacher_path']) for r in rows}
paths|={Path(r['margin_target_registry']) for r in rows}
paths|={Path(r['classification_target_registry']) for r in rows}
new_json(OUT/'manifest/frozen_execution.json',dict(
    schema='strict_v5.lowshot_depth_noise_3x3.reference_fix.frozen.v2',
    sealed_at_utc=now(),files=[source_record(p) for p in sorted(paths)]))
print(dict(frozen_sha256=digest(OUT/'manifest/frozen_execution.json'),files=len(paths)))
