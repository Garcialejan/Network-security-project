'''
This setup.py file is an essential part of packaging and 
distributing the project. It is used by setuptools (or distutils
in older Python versions) to define the configuration of the project,
such as its metadata, dependencies, and more
'''

from setuptools import find_packages,setup
from typing import List

def get_requirements()->List[str]:
    """
    This function will return list of requirements
    
    """
    requirement_lst:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            # Read lines from the file
            lines=file.readlines()
            # Process each line
            for line in lines:
                requirement=line.strip()
                ## ignore empty lines and -e .
                if requirement and requirement!= '-e .':
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")
    return requirement_lst

# Install automatically all the requirements needed
setup(
    name="NetworkSecurity",
    version="0.0.1",
    author="Alejandro García",
    author_email="garcialejan@gmail.com",
    packages=find_packages(), # Searching all the packages
    install_requires=get_requirements() # Install requirements when the packages getting build
)

#! We use -e to trigger the setup.py file and install dependencies in editable 
# mode. The "." reflects the actual directory.


# -e : This is an option in pip that stands for ‘editable’. 
# When you use this option, pip installs the package in 
# editable mode. This allows changes made to the source 
# code of the package (in the local directory) to be 
# immediately reflected without the need to reinstall the package.
#* It will create a symbolic link between the virtual environment and
#* the project directory, allowing any changes to the source code to be
#* automatically reflected in the environment without the need to reinstall
#* the package.