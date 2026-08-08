from setuptools import find_packages, setup


setup(
    name="differentiable-dag-sampling-jax",
    version="0.1",
    description="Differentiable DAG Sampling in JAX",
    author="Anonymous",
    author_email="anonymous@mail.com",
    packages=find_packages(),
    install_requires=[
        "jax",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "networkx",
    ],
    zip_safe=False,
)

