# setup.py
# https://github.com/ddh0/easy-llama/
__version__ = '0.1.113'

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
    license='The Unlicense',
    include_package_data=True,
    packages=['easy_llama', 'easy_llama.assets'],
    package_data={
        'easy_llama.assets': [
            '*.ico',
            '*.png',
            '*.html',
            '*.css',
            '*.js',
            '*.webmanifest'
        ]
    },
    install_requires=[
        'llama_cpp_python',
        'numpy',
        'Flask',
        'cryptography'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Framework :: Flask',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: Public Domain',  
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
