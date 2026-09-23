from setuptools import find_packages, setup

setup(
    name="app-planner",
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "app-planner = app_planner.cli:cli_main",
        ],
    },
)
