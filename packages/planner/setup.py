from setuptools import find_packages, setup

setup(
    name="planner",
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # Library-only dist: it exists to provide planner.engine (the transition
    # state machine grib's GUI and kanban modules import). The CLI is grib's —
    # declaring `planner = grib.cli:main` here made two distributions ship the
    # same command, and the loser of that collision is what broke the install.
)
