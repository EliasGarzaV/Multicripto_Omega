from setuptools import find_packages, setup

setup(
    name="pipeline_rh",
    packages=find_packages(exclude=["pipeline_rh_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud",
        'scikit-learn',
        'pandas',
        'fastparquet',
        'pyodbc'
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
