from setuptools import setup, find_packages

setup(
    name="app-planner",
    version="0.2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "planner = app_planner.core:cli_main",
            "notes = app_planner.core:cli_main",
            "kanban = app_planner.kanban:cli_main",
            "sync-notes-to-board = app_planner.sync:cli_main",
        ],
    },
)
