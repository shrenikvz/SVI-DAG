from setuptools import find_packages, setup


setup(
    name="bayesdag-jax",
    version="0.1",
    description="JAX port of BayesDAG for continuous benchmark settings",
    author="Anonymous",
    author_email="anonymous@mail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=["jax", "numpy", "scipy"],
    zip_safe=False,
)

