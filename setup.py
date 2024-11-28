import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="BraidAnalysis", 
    version="0.0.1",
    author="Floris van Breugel, David Stupski, Gael Robb",
    author_email="fvanbreugel@unr.edu",
    description="Standard analysis functions for braid data and flies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)