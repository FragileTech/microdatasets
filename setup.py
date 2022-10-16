"""microdatasets package installation metadata."""
from importlib.machinery import SourceFileLoader
from pathlib import Path

from setuptools import find_packages, setup


version = SourceFileLoader(
    "microdatasets.version",
    str(Path(__file__).parent / "src" / "microdatasets" / "version.py"),
).load_module()

with open(Path(__file__).with_name("README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="microdatasets",
    description="Simple datasets for evaluating DL algorithms.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    version=version.__version__,
    license="MIT",
    author="Guillem Duran Ballester",
    author_email="guillem@fragile.tech",
    url="https://github.com/FragileTech/microdatasets",
    keywords=["Machine learning", "artificial intelligence"],
    test_suite="tests",
    tests_require=["pytest>=5.3.5", "hypothesis>=5.6.0"],
    extras_require={},
    install_requires=["matplotlib", "numpy"],
    package_data={
        "": ["README.md"],
        "microdatasets": ["assets/**/*", "assets/**/.*", "tests/**/*", "tests/**/.*"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries",
    ],
)
