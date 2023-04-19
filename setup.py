from setuptools import setup, find_packages

VERSION = '0.1' 
DESCRIPTION = 'Generative Image2Text with Segment Anything Model'
LONG_DESCRIPTION = 'My first Python package with a slightly longer description'

# Setting up
setup(
        name="samgit", 
        version=VERSION,
        author="Jimmy Yao",
        author_email="jinghaoyao@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=open("requirements.txt").read().splitlines(), # add any additional packages that 
)