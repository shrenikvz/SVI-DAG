from setuptools import find_packages, setup


setup(
    name="prodag-jax",
    version="0.1",
    description="JAX port of ProDAG",
    author="Anonymous",
    author_email="anonymous@mail.com",
    packages=find_packages(),
    install_requires=["jax", "numpy", "networkx"],
    zip_safe=False,
)

