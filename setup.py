from setuptools import find_packages, setup

_core = "packages/core-py"

# The consolidation moved domain/ to the repo root, so
# find_packages(where=_core) -> [] while pyproject's packages.find
# discovers domain at "." — package_dir must map that root explicitly or
# setuptools resolves 'domain' to packages/core-py/domain and egg_info dies
# (card 56cf354c: "package directory ... does not exist"). apps.cli.sloughgpt
# is stale: apps/cli has no sloughgpt/ package; the console entry point is
# apps.cli.src.cli:main.
setup(
    packages=find_packages(where=_core, include=("domain*", "utils*"))
    + find_packages(where=".", include=("domain*",))
    + ["apps.cli"],
    package_dir={
        "": _core,
        "apps.cli": "apps/cli",
        "domain": "domain",
    },
)
