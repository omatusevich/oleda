import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Declared dependencies are the packages oleda imports DIRECTLY, with lower bounds rather than
# exact pins. A library pinning exact versions - as this did, by reading the pip-freeze in
# requirements.txt - forces its whole development environment on anything that installs it and
# cannot coexist with other packages. Transitive dependencies (scikit-learn, numba, joblib,
# cloudpickle, ...) arrive via shap and lightgbm and are deliberately not listed; notebook
# tooling (ipykernel, debugpy, jupyter_client, pyzmq, tornado) and the macOS-only appnope are
# not runtime requirements at all.
#
# requirements.txt remains the pinned, reproducible DEVELOPMENT environment.
INSTALL_REQUIRES = [
    "pandas>=1.3",
    "numpy>=1.20",
    "scipy>=1.7",
    "matplotlib>=3.4",
    "seaborn>=0.12",      # histplot, pairplot(corner=)
    "shap>=0.40",         # TreeExplainer
    "lightgbm>=3.3",      # LGBMClassifier / LGBMRegressor
    "statsmodels>=0.13",  # ols, stats.multicomp
    "ipython>=7.0",       # IPython.display is used to render the report
]

setuptools.setup(
    name="oleda",
    version="1.0.0",
    author="omatusevitch",
    author_email="omatusevitch@gmail.com",
    description=("Automated exploratory data analysis for pandas DataFrames: "
                 "single-dataset and two-dataset comparison reports with SHAP "
                 "feature ranking, Cramer's V and ANOVA"),
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
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=setuptools.find_packages(exclude=["tests", "tests.*"]),
    # python_requires now matches the classifiers above; the previous >=3.6 was inconsistent
    # with them and with the pinned numpy, which needs 3.9+.
    python_requires=">=3.9",
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    install_requires=INSTALL_REQUIRES,
)
