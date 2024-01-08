# setup.py we can use to build the application as a package itself and deployed as a package.

from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."  # from the requirements file, so that it indicates the end of the file and it says setup.py is recognized

def get_requirements(filename:str)->List[str]:
    
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(filename) as file_handle:
        requirements = file_handle.readlines()
        requirements = [requirement.replace("\n","") for requirement in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
name = "mlproject",
version = "0.0.1",
author = "KeerthiRaj",
author_email = "keerthirajkv2@gmail.com",
packages = find_packages(),
install_requires = get_requirements("requirements.txt")
)