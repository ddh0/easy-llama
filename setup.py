# setup.py
# https://github.com/ddh0/easy-llama/
# MIT License -- Copyright (c) 2024 Dylan Halladay

__version__ = '0.2.0'

from setuptools import setup


setup(
    name='easy_llama',
    version=__version__,
    description='Text generation in Python, as easy as possible',
    long_description="For more information, visit the project's GitHub repository:\n\nhttps://github.com/ddh0/easy-llama",
    long_description_content_type='text/markdown',
    url='https://github.com/ddh0/easy-llama/',
    author='Dylan Halladay',
    author_email='dylanhalladay02@icloud.com',
    license='MIT',
    include_package_data=True,
    packages=['easy_llama', 'easy_llama.webui'],
    package_data={
        'easy_llama.webui': [
            '*.ico',
            '*.png',
            '*.html',
            '*.css',
            '*.js',
            '*.webmanifest'
        ]
    },
    install_requires=[
        'numpy',
        'fastapi',
        'uvicorn',
        'jinja2'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',
        'Natural Language :: English',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
