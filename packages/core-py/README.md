## Core Python (`domains`)

`packages/core-py/` is on the Python path as the **`domains`** package (and related **`utils`**) when you install from the repo root (**`python3 -m pip install -e ".[dev]"`**).

Training, inference, models, and infrastructure code live under **`domains/`**. The API imports these modules; keep heavy logic here instead of in **`apps/api/server/`** route handlers. Trainer-native **`step_*.pt`** bundles embed **`stoi` / `itos` / `chars`** for char-LM eval; see **`docs/policies/CONTRIBUTING.md`** (*Checkpoint vocabulary*).

See **docs/STRUCTURE.md** and **docs/AI_SOFTWARE_ENGINEERING.md**.
