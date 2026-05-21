from setuptools import find_packages, setup

_core = "packages/core-py"

setup(
    packages=find_packages(where=_core, include=("domains*", "utils*"))
    + ["apps.cli", "apps.cli.sloughgpt"],
    package_dir={"": _core, "apps.cli": "apps/cli"},
)
