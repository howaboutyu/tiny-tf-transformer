from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="tiny-tf-transformer",
    version="0.0.0",
    author="Haobo Yu",
    author_email="haoboyu806@gmail.com",
    description="A tiny transformer library in TensorFlow 2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/howaboutyu/tiny-tf-transformer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        # Add any dependencies required by SynapseFlow here
        "tensorflow",
        "tensorflow_datasets",
        "tensorflow_text",
        "numpy",
        "tqdm",
        "pytest",
    ],
    python_requires=">=3.6",
)
