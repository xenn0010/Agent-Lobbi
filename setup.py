#!/usr/bin/env python3
"""
Setup script for agent-lobbi - Enhanced A2A+ Agent Collaboration Platform
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements
def read_requirements(filename):
    """Read requirements from file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        return []

requirements = read_requirements('requirements.txt')

# Version information
VERSION = "1.0.1"

setup(
    name="agent-lobbi",
    version=VERSION,
    author="Agent Lobby Team",
    author_email="support@agentlobby.com",
    description="Enhanced A2A+ Agent Collaboration Platform with Advanced Metrics and Neuromorphic Intelligence",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/agentlobby/agent-lobbi",
    project_urls={
        "Homepage": "https://agentlobby.com",
        "Documentation": "https://docs.agentlobby.com",
        "Source": "https://github.com/agentlobby/agent-lobbi",
        "Bug Tracker": "https://github.com/agentlobby/agent-lobbi/issues",
        "Changelog": "https://github.com/agentlobby/agent-lobbi/blob/main/CHANGELOG.md"
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
        "Topic :: System :: Networking",
        "Framework :: AsyncIO",
        "Environment :: Web Environment",
        "Typing :: Typed"
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0"
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0"
        ],
        "monitoring": [
            "prometheus-client>=0.15.0",
            "grafana-client>=3.0.0"
        ],
        "enterprise": [
            "redis>=4.0.0",
            "celery>=5.0.0",
            "kubernetes>=24.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "agent-lobbi=src.main:main",
            "agent-lobbi-server=src.core.lobby:main",
            "agent-lobbi-metrics=src.core.agent_metrics_enhanced:main"
        ]
    },
    include_package_data=True,
    package_data={
        "src": ["*.yaml", "*.yml", "*.json", "*.toml"],
        "src.config": ["*.yaml", "*.yml", "*.json"],
        "src.templates": ["*.html", "*.jinja2"],
        "src.static": ["*.css", "*.js", "*.png", "*.jpg", "*.svg"]
    },
    zip_safe=False,
    keywords=[
        "agent",
        "collaboration",
        "a2a",
        "artificial intelligence",
        "multi-agent systems",
        "distributed computing",
        "neuromorphic",
        "metrics",
        "analytics",
        "websocket",
        "asyncio",
        "real-time",
        "business intelligence"
    ],
    platforms=["any"],
    license="MIT",
    
    # Enhanced metadata for PyPI
    maintainer="Agent Lobby Team",
    maintainer_email="maintainers@agentlobby.com",
    
    # Advanced features
    cmdclass={},
    
    # Security and compliance
    download_url=f"https://github.com/agentlobby/agent-lobbi/archive/v{VERSION}.tar.gz",
    
    # PyPI upload configuration
    options={
        "bdist_wheel": {
            "universal": False,
            "plat_name": "any"
        }
    }
) 