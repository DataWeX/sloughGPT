from setuptools import find_packages, setup

setup(
    name="planner",
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            # The CLI consolidated into app_planner.cli; planner.cli was
            # deleted with it. Both dists now ship the same real CLI.
            "planner = app_planner.cli:main",
        ],
    },
)
