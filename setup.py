from setuptools import setup, find_packages

setup(
    name='safe_transfer',
    version='1.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        "torch",
        "numpy",
        "gymnasium<1.0",
        "pygame",
        "pymunk",
        "tyro",
        "pandas",
        "tensorboard",
        # "wandb"  # For online logging
    ],
)
