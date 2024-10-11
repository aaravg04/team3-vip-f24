from setuptools import setup
import os
from glob import glob

package_name = 'reactive_racing'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jacob Knaup',
    maintainer_email='jacobk@gatech.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception = reactive_racing.perceive_boundaries:main',
            'planning = reactive_racing.estimate_centerline:main',
            'control = reactive_racing.compute_control:main',
            'zed2_custom_node = reactive_racing.zed2_custom_node:main',
        ],
    },
)