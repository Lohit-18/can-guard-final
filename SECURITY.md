# Security policy

## What this project is, and is not

CAN-Guard is a **research and teaching project**. It detects attacks on
*simulated* CAN traffic that it generates itself. It has not been tested on a
real vehicle, has not been through any automotive safety process, and is not
certified against ISO 21434, ISO 26262 or UN R155.

**Do not connect this to a vehicle you intend to drive**, and do not rely on it
as a safety control. If you adapt it for real hardware, treat it as a monitoring
aid that raises alarms - never as something that may modify, block or inject bus
traffic.

## Nothing here is an attack tool

The code in `canguard/attacks.py` generates labelled attack traffic *inside the
simulator* so the detector has something to learn from. It does not talk to a
CAN interface, it does not open a socket, and it produces CSV rows, not frames on
a wire. Pull requests that turn it into something that transmits on real hardware
will be declined.

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | Yes |
| < 1.0 | No |

## Reporting a vulnerability

If you find a security problem **in this code** - for example a path traversal in
the trace loader, unsafe deserialisation of a model file, or a dependency with a
known CVE that this project pins - please report it privately rather than opening
a public issue:

1. Open a [private security advisory](https://github.com/yourname/can-guard/security/advisories/new)
   on GitHub, or
2. email the maintainer address listed on the repository profile.

Please include the version, the steps to reproduce, and what an attacker would
gain. You can expect an acknowledgement within **7 days** and an assessment
within **30 days**. If the report is valid, you will be credited in the
changelog unless you would rather not be.

## Known risk areas worth knowing about

These are inherent to the design rather than bugs, and are documented so you can
judge them for yourself:

- **Model files are pickles.** `models/*.joblib` are loaded with `joblib.load`,
  which executes arbitrary code by design. Only load model files you generated
  yourself or obtained from a source you trust. Re-running `python run_all.py`
  rebuilds all of them locally in about two minutes, which is the safe default.
- **Keras files are code-adjacent.** `.keras` archives can carry custom objects.
  The same rule applies.
- **Traces are untrusted input.** `canguard/dataset.load_trace` parses a CSV
  from disk. It validates shape but assumes the file is well-formed; feeding it
  a hostile multi-gigabyte file is a denial-of-service against your own process.
- **No network code exists in this project.** If a future version adds any, this
  section should be revised before it merges.
