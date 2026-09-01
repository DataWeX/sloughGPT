from setuptools import setup, find_packages

setup(
    name="planner",
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "planner = planner.cli:main",
        ],
    },
)
