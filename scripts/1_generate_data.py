"""
Step 1 - generate the CAN traces.

Three traces are produced:

  baseline.csv  attack-free.   Used to teach the unsupervised model what a
                               healthy bus looks like. Never contains attacks.
  train.csv     four attack families. Used to fit the supervised models.
  test.csv      five attack families - the fifth (`replay`) is deliberately
                held out of training so we can measure how the detector copes
                with an attack it has never seen.
"""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from canguard import attacks, dataset, simulator

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"

TRAIN_ATTACKS = ["throttle_spoof", "fake_brake", "dos_flood", "fuzzing"]
TEST_ATTACKS = TRAIN_ATTACKS + ["replay"]          # replay = unseen at train time


def build(name, duration, seed, kinds, episodes):
    t0 = time.time()
    frames, states = simulator.simulate(duration_s=duration, seed=seed)
    if kinds:
        frames, log = attacks.apply_attacks(frames, states, seed=seed + 500,
                                            duration=duration, kinds=kinds,
                                            episodes_per_kind=episodes)
        dataset.save_episode_log(log, DATA / f"{name}_episodes.csv")
    else:
        log = []
    dataset.save_trace(frames, DATA / f"{name}.csv")
    n_att = sum(1 for f in frames if f[3] == 1)
    print(f"  {name:9s} {duration:5.0f}s  {len(frames):>8,} frames  "
          f"{n_att:>7,} attack frames ({100*n_att/len(frames):5.2f}%)  "
          f"{len(log):>2} episodes   [{time.time()-t0:.1f}s]")


if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    print("Generating CAN traces...")
    build("baseline", 360.0, 11, [], 0)
    build("train",    600.0, 22, TRAIN_ATTACKS, 7)
    build("test",     420.0, 33, TEST_ATTACKS, 5)
    print(f"\nWritten to {DATA}")
