"""
SloughGPT SDK - Python Package Setup
"""

import os
from setuptools import setup, find_packages

readme_path = os.path.join(os.path.dirname(__file__), "README.md")
with open(readme_path, "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sloughgpt-sdk",
    version="1.0.0",
    author="SloughGPT",
    author_email="dev@sloughgpt.ai",
    description="Python SDK for SloughGPT API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iamtowbee/sloughGPT",
    project_urls={
        "Bug Tracker": "https://github.com/iamtowbee/sloughGPT/issues",
        "Documentation": "https://github.com/iamtowbee/sloughGPT#readme",
        "Source Code": "https://github.com/iamtowbee/sloughGPT",
    },
    packages=find_packages(exclude=["tests", "tests.*", "scripts", "scripts.*"]),
    package_data={
        "sloughgpt_sdk": ["py.typed"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "httpx>=0.22.0",
            "black>=22.0.0",
            "mypy>=0.950",
            "ruff>=0.1.0",
        ],
        "async": [
            "httpx>=0.22.0",
        ],
        "websocket": [
            "websocket-client>=1.0.0",
        ],
        "all": [
            "httpx>=0.22.0",
            "websocket-client>=1.0.0",
            "pytest>=7.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0.0",
            "mypy>=0.950",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sloughgpt-cli=sloughgpt_sdk.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
