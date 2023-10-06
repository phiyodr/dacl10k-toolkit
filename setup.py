import setuptools

print(setuptools.find_packages())


with open("README.md", "r") as file:
    long_description = file.read()

with open("requirements.txt") as file:
    required = file.read().splitlines()
    
setuptools.setup(
    name="dacl10k",
    version="0.3",
    author="Philipp J. Roesch",
    author_email="phiyodr@gmail.com",
    description="dacl10k-toolkit for dacl10k Challenge at WACV2024",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/phiyodr/dacl10k-toolkit",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    package_data={
        # If any package contains *.txt or *.rst files, include them:
        "": ["*.txt", "*.csv", "*.json"]},
    install_requires=[
            "torch", 
            "torchvision",
            "scikit-learn",
            "pandas",
            "numpy",
            "matplotlib",
            "shapely",
            "scikit-image",
            "pillow"
        ]
    )
