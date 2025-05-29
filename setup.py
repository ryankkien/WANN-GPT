from setuptools import setup, find_packages

setup(
    name="wann-gpt",
    version="0.1.0", 
    description="Weight-Agnostic Neural Networks applied to GPT-2 Transformers",
    author="Research Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "scipy",
        "matplotlib",
        "transformers>=4.20.0",
        "datasets",
        "tokenizers",
        "tqdm",
        "wandb",
        "numba",
        "cupy-cuda12x",
        "scikit-learn",
        "pandas",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Researchers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 