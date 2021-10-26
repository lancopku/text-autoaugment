import os
import pkg_resources
from setuptools import setup, find_packages

setup(
    name="taa",
    version="1.00",
    description="augment your dataset with text auto augmentation",
    author="Shuhuai Ren, Yuxiang Wang",
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    
)