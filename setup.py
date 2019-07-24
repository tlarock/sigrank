import setuptools


#with open('requirements.txt') as file:
#    requires = [line.strip() for line in file if not line.startswith('#')]

setuptools.setup(
    name='sigrank',
    version='0.0.1',
    author='Tim LaRock',
    author_email='timothylarock@gmail.com',
    description='Tools for ranking significant episodes.',
    url='https://github.com/tlarock/sigrank.git',
    packages=setuptools.find_packages(),
    #install_requires=requires,
    classifiers=['Programming Language :: Python :: 3',
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: OS Independent'],
)
