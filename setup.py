#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_namespace_packages, Extension
import os

with open("README.rst") as readme_file:
    readme = readme_file.read()

# The requirements section should be kept in sync with the environment.yml file
requirements = [
    # fmt: off
    "click>=7.0",
    "click-plugins",
    "entrypoints",
    "numpy",
    "pillow",
    # fmt: on
]

setup_requirements = [
    # fmt: off
    "cython",
    "numpy",
    "setuptools_scm",
    "setuptools_scm_git_archive",
    "wheel",
    # fmt: on
]

test_requirements = [
    # fmt: off
    "pytest>=3",
    "pytest-cov",
    # fmt: on
]

doc_requirements = [
    # fmt: off
    "sphinx",
    "sphinx-autoapi",
    "sphinx-click",
    "watchdog",
    # fmt: on
]

conda_requires = {
    # fmt: off
    "build": [
    ],
    "host": [
        "darknet-cpu >=0.0.20200614"
    ],
    "run": [
        "darknet >=0.0.20200614"
    ]
    # fmt: on
}

conda_rosetta_stone = {
    # fmt: off
    "pypa-requirement": "conda-dependency"
    # fmt: on
}

ext_modules = []
try:
    from Cython.Build import cythonize
    import numpy as np
    ext_modules = cythonize([
        Extension("darknet.py.network", ["src/darknet/py/network.pyx"],
                  include_dirs=[np.get_include()],
                  libraries=["darknet"])
    ])
except ModuleNotFoundError:
    pass

setup_kwargs = dict(
    author="Patrick Sodré",
    author_email="psodre@gmail.com",
    use_scm_version={"write_to": "src/darknet/py/_version.py"},
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="DarkNet OpenSource Neural Networks in Python",
    # fmt: off
    entry_points={
        "darknet.cli": [
            "py=darknet.py.cli:py",
        ],
    },
    # fmt: on
    ext_modules=ext_modules,
    install_requires=requirements,
    license="MIT",
    long_description=readme,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    keywords="py darknet",
    name="darknet-py",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="./src"),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    extras_require={
        # fmt: off
        "test": test_requirements,
        "doc": doc_requirements,
        # fmt: on
    },
    url="https://github.com/zeroae/darknet.py",
    zip_safe=False,
)

if "CONDA_BUILD_STATE" in os.environ:
    try:
        from setuptools_scm import get_version

        setup_kwargs["version"] = get_version(**setup_kwargs["use_scm_version"])
        del setup_kwargs["use_scm_version"]
    except ModuleNotFoundError:
        print(
            "Error: darknet.py requires that setuptools_scm be installed with conda-build!"  # noqa: E501
        )
        raise
    setup_kwargs["conda_rosetta_stone"] = conda_rosetta_stone
    setup_kwargs["conda_requires"] = conda_requires

setup(**setup_kwargs)
