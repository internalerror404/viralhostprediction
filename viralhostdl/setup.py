from setuptools import setup, find_packages

setup(
    name='viralhost',
    version='0.667',
    author='Hina Dixit',
    py_modules=['viralhost'],
    packages=find_packages(),
    package_data={'viralhost': ['weights/*.pkl']},
    # include_package_data=True,
    install_requires=[
        'Click',
        'numpy',
        'keras',
        'sklearn'
    ],
    entry_points={
        'console_scripts': [
            'viralhost =viralhost:cli'
        ]}

)

