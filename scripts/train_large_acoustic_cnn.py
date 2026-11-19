"""Run the larger acoustic CNN study as a resumable, bounded worker."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
for key, value in {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2",
                   "MPLCONFIGDIR": "/tmp/mimii-matplotlib"}.items():
    os.environ.setdefault(key, value)

from mimii.ae_experiment import write_json
from mimii.large_cnn_experiment import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_large_acoustic_cnn.json")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    study = json.loads(Path(args.config).read_text())
    root = Path(study["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    if args.detach:
        with (root / "training.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), "--config", str(Path(args.config).resolve())],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(root / "worker.json", {"pid": process.pid})
        print(f"Started larger CNN PID {process.pid}; log: {root / 'training.log'}")
    else:
        with (root / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                run(study)
            except TimeoutError as error:
                write_json(root / "status.json", {"state": "budget_exhausted", "reason": str(error)})
            except BaseException as error:
                write_json(root / "status.json", {"state": "failed", "error": repr(error)})
                raise
