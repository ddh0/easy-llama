# easy_llama.py
# https://github.com/ddh0/easy-llama/

from setuptools import setup

setup(
    name='easy_llama',
    version='0.1.8',
    description='Text generation in Python, made easy',
    long_description="""To view the current documentation for easy-llama, visit the project's GitHub repository:\nhttps://github.com/ddh0/easy-llama""",
    url='https://github.com/ddh0/easy-llama/',
    author='Dylan Halladay',
    author_email='dylanhalladay02@icloud.com',
    license='The Unlicense',
    packages=['easy_llama'],
    install_requires=[
        'llama_cpp_python',
        'colorama',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: Public Domain',  
        'Operating System :: POSIX :: Linux',
        'Natural Language :: English',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)
