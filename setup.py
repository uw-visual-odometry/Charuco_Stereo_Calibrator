from setuptools import setup, find_packages

# Read the content of your README file
with open("README.md", encoding="utf-8") as readme_file:
    long_description = readme_file.read()

setup(
    name="Charuco_Stereo_Calibrator",  # Package name
    version="0.1.0",  # Initial release version
    author="Kim, Huijo",
    author_email="ccomkhj@gmail.com",
    description="A package for calibrating stereo cameras using Charuco boards.",
    long_description=long_description,
    long_description_content_type="text/markdown",  # This indicates using markdown for your long description
    url="https://github.com/ccomkhj/Charuco_Stereo_Calibrator",  # URL to the repository or project site
    packages=find_packages(
        include=["csc", "csc.*"]
    ),  # Only include the packages within the 'csc' directory
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Choose the appropriate license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",  # Specify the Python versions you support
    install_requires=[  # Dependencies are listed in requirements.txt, so read them
        line.strip() for line in open("requirements.txt").readlines()
    ],
)
