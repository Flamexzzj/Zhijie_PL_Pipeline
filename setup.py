from setuptools import setup, find_packages

setup(
    # this sets the package name that will be installed
    name="PL_Support_Codes",
    version="0.0.1",
    install_requires=[
        "requests>=2.25.0,<3.0.0",
        "toml>=0.10.0,<1.0.0",
        'importlib-metadata>=4.0.0; python_version > "3.9"',
        "hydra-core>=1.2.0,<2.0.0",
        "opencv-python>=4.5.0,<5.0.0",
        "imagecodecs>=2021.0.0",
        "yapf>=0.30.0",
        "pytorch_lightning>=1.8.0,<2.0.0",
        "torch>=1.12.0,<2.0.0",
        "torchvision>=0.13.0,<1.0.0",
        "torchmetrics>=0.9.0,<1.0.0",
        "numpy>=1.21.0,<2.0.0",
        "einops>=0.6.0",
        "rasterio>=1.2.0",
        "geopandas>=0.10.0",
        "tqdm>=4.60.0"
    ],
    # this add the __init__.py files in the package directories so that code can call them 
    packages=find_packages(
        # All keyword arguments below are optional:
        where=".",  # '.' by default
        include=["PL_Support_Codes"],  # ['*'] by default
        exclude=["st_water_seg.egg-info", "dist",
                 ".vscode"],  # empty by default
    ),
)
