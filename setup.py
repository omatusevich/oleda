import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

INSTALL_REQUIRES = []
with open("requirements.txt", encoding="utf-8") as req:
    for line in req.readlines():
        INSTALL_REQUIRES.append(line.split("#")[0].strip())

setuptools.setup(
    name="oleda",
    version="1.0.0",
    author="omatusevitch",
    author_email="omatusevitch@gmail.com",
    description="an eda",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Banuba/oleda",
    project_urls={
        "Bug Tracker": "https://github.com/Banuba/oleda/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,# True will install all files in repo
    install_requires=INSTALL_REQUIRES,
    python_requires=">=3.6",
)
