from setuptools import setup

setup(
    name="dev-notes",
    version="0.1.0",
    description="Standalone development journal — file-backed notes with YAML frontmatter",
    packages=["notes"],
    package_dir={"notes": "."},
    entry_points={
        "console_scripts": [
            "notes = notes.notes:cli_main",
        ],
    },
    python_requires=">=3.9",
)
