from setuptools import setup, find_packages

setup(
    name="dtc",
    version="0.1",
    packages=["model_trainer", "model_loader", "models"],
    package_dir={
        "": ".",
        "model_loader": "./model_loader",
        "model_trainer": "./model_trainer",
        "models": "./models",
    },
)
