import setuptools

# Read in the requirements.txt file
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Define the setup configuration
setuptools.setup(
    name="mimesys",
    version="0.1.0",
    author="Donghyun Kim, Zichao Hu",
    author_email="donghyun@utexas.edu, zichao@utexas.edu",
    description="mimesys",
    packages=setuptools.find_packages(include=["mimesys"]),
    install_requires=requirements,
    python_requires=">=3.9",
)
