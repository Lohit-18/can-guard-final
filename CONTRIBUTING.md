# Contributing to CAN-Guard

Thanks for taking a look. This is a small, focused project, so the bar is
simple: a change should make the detector more honest, more realistic, or
easier to run.

## Getting set up

```bash
git clone https://github.com/yourname/can-guard.git
cd can-guard
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pytest               # for the test suite
python run_all.py                # confirm the baseline reproduces
pytest -q
```

If `run_all.py` prints the metrics in the README, your environment is good.

## The one rule that matters

**Do not make the problem easier by accident.**

This is the failure mode that quietly ruins intrusion-detection projects. An
attack that is trivially detectable produces a beautiful F1 score and teaches
nothing. Two examples from this repository's own history:

- The spoofing attacks originally re-randomised their payload every frame, which
  made them detectable from jitter alone. They now ramp smoothly.
- The brake spoof originally never touched the brake-light bit, which made the
  cross-ECU check a giveaway. Half the simulated attackers now spoof it too.

So if your change moves a metric *up*, say why in the pull request, and be
specific about which signal the models are now using. A jump from 0.92 to 0.99
usually means a leak, not a breakthrough.

## Good contributions

- **New attack families.** Bus-off attacks, diagnostic-session abuse, gateway
  spoofing, timing side channels. Add them to `canguard/attacks.py` and give
  them an entry in `ATTACK_TYPES`.
- **Harder versions of existing attacks.** Lower intensity, shorter episodes,
  attackers that time their injection into the genuine frame's jitter window.
- **Features grounded in something real.** A feature should correspond to a
  property of the bus or the vehicle you can name in one sentence.
- **Real-data adapters.** A loader that converts a `candump` log or a public
  CAN-IDS dataset into the CSV shape this pipeline reads.
- **Documentation of what fails.** Negative results are welcome and go in the
  README's limitations section.

## Less useful

- Swapping the models for bigger ones without a measured reason. The bottleneck
  here is the feature set and the realism of the traffic, not model capacity.
- Adding dependencies. Six packages, all permissively licensed, is the budget.
- Reformatting passes that touch files unrelated to the change.

## Before you open a pull request

1. `pytest -q` passes.
2. `python run_all.py` completes and you have noted any metric that moved.
3. New code has comments that explain *why*, not *what*. The existing files set
   the tone - every non-obvious choice says what it is defending against.
4. Lines stay under 88 characters, four-space indentation, no trailing
   whitespace.

## Reporting a problem

Use the issue templates. For anything that looks like a security problem in the
code itself, read [SECURITY.md](SECURITY.md) first.

## Licence

By contributing you agree that your contribution is licensed under the MIT
licence that covers this repository.
