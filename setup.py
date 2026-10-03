from setuptools import find_packages, setup

_core = "packages/core-py"

setup(
    packages=find_packages(where=_core, include=("domains*", "utils*"))
    # `domain/`, `services/`, `testing/` and `apps.cli` live at repo root,
    # outside `packages/core-py`, so the default "" mapping would resolve them
    # to packages/core-py/<name> (nonexistent) and they would never be
    # published. Publish each one explicitly.
    + find_packages(where=".", include=("domain*", "services*", "testing*"))
    + ["apps.cli", "apps.cli.sloughgpt"],
    package_dir={
        "": _core,
        "domain": "domain",
        "services": "services",
        "testing": "testing",
        "apps.cli": "apps/cli",
    },
)
