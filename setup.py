from setuptools import setup, find_packages

setup(
    name="simple-pytorch-lora",
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="A lightweight, library-agnostic LoRA implementation for PyTorch",
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/YourUsername/Simple-PyTorch-LoRA",
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.8.0',
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'simple-pytorch-lora=simple_pytorch_lora.__main__:main',
        ],
    },
)
