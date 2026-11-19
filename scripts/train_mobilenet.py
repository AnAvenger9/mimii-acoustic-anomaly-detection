"""Run/resume the frozen MobileNet baseline; keep all outputs inside the project."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))
for key, value in {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2",
                   "MPLCONFIGDIR": "/tmp/mimii-matplotlib", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}.items():
    os.environ.setdefault(key, value)

from mimii.finetune import FinetuneConfig, benchmark, run


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_mobilenet_v3_small.json")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    config = FinetuneConfig(**json.loads(Path(args.config).read_text()))
    if args.benchmark:
        benchmark(config)
        return
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "training.log").open("a") as handle:
        with contextlib.redirect_stdout(Tee(sys.stdout, handle)), contextlib.redirect_stderr(Tee(sys.stderr, handle)):
            try:
                run(config)
            except BaseException as error:
                (output / "status.json").write_text(json.dumps({"state": "failed", "error": repr(error)}) + "\n")
                raise


if __name__ == "__main__":
    main()
