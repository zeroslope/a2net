from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    "glog",
    "tensorlayer==1.11.1",
    "easydict"
]

setup(
    name='trainer',
    version='0.2',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My loss function.'
)
