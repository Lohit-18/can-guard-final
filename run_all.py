#!/usr/bin/env python3
"""
Run the whole CAN-Guard pipeline end to end:

    python run_all.py

Takes about three to five minutes on a normal laptop. No GPU, no downloads,
no accounts, no paid services.
"""

import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
STEPS = [
    ("Generating CAN traces",        "scripts/1_generate_data.py"),
    ("Extracting features",          "scripts/2_extract_features.py"),
    ("Training the models",          "scripts/3_train.py"),
    ("Evaluating on unseen traffic", "scripts/4_evaluate.py"),
]

if __name__ == "__main__":
    t0 = time.time()
    for i, (title, script) in enumerate(STEPS, 1):
        print(f"\n{'='*78}\n  STEP {i}/{len(STEPS)}  -  {title}\n{'='*78}")
        r = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"\nStep {i} failed. Fix the error above and re-run.")
    print(f"\nDone in {time.time()-t0:.0f}s.")
    print(f"Charts and metrics: {ROOT / 'reports'}")
    print("Live demo:          python scripts/5_live_detect.py")
