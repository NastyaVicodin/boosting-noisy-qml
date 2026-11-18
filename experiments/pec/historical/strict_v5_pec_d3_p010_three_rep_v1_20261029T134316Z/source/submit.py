"""One-shot Slurm submission with append-only intent and receipt; not run by preparation."""
from common import *
from launch_preflight import preflight
import subprocess


def submit():
    intent = OUT / "manifest/submission_intent.json"
    receipt_path = OUT / "manifest/submission_receipt.json"
    assert not intent.exists() and not receipt_path.exists()
    assert not any((OUT / "runs").iterdir())
    checked = preflight()
    script = OUT / "launch_pec_r02.sbatch"
    command = ["sbatch", "--parsable", str(script)]
    new_json(intent, {"utc": now(), "command": command,
                      "script_sha256": digest(script), "preflight": checked,
                      "user_authorized": True})
    result = subprocess.run(command, capture_output=True, text=True)
    receipt = {"utc": now(), "command": command,
               "returncode": result.returncode,
               "stdout": result.stdout, "stderr": result.stderr,
               "protocol_sha256": digest(OUT / "manifest/protocol.json"),
               "frozen_sha256": digest(OUT / "manifest/frozen_execution.json")}
    if result.returncode == 0:
        receipt["job_id"] = result.stdout.strip().split(";")[0]
    new_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True))
    if result.returncode:
        raise RuntimeError("Slurm submission failed; no automatic retry")


if __name__ == "__main__":
    submit()
