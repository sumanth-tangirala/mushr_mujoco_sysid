from pathlib import Path

from setuptools import find_packages, setup


def read_long_description() -> str:
    readme = Path(__file__).parent / "README.md"
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return "MuSHR MuJoCo system identification models and training utilities."


setup(
    name="mushr_mujoco_sysid",
    version="0.1.0",
    description="System identification models and training pipeline for MuSHR MuJoCo.",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    author="",
    packages=find_packages(
        exclude=("env", "experiments", "docs", "scripts", "configs", "data")
    ),
    python_requires=">=3.8",
    install_requires=[],
)
