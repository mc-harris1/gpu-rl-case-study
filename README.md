# gpu-rl-case-study

## Legal note (Atari ROMs)

No ROMs are included in this repo. Install Atari ROMs via the official Gymnasium/ALE flow (license acceptance required).

## Install (uv)

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[atari]
AutoROM --accept-license
```

## Strategy

Code evolves by first providing deterministic replay + telemetry foundation.
