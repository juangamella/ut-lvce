import setuptools

setuptools.setup(
    name='utlvce',
    version='0.1.0',
    author='Juan L Gamella',
    author_email='juangamella@gmail.com',
    packages=['utlvce'],
    scripts=[],
    url='https://github.com/juangamella/ut-lvce',
    license='BSD 3-Clause License',
    description='',
    long_description=open('README_pypi.md').read(),
    long_description_content_type="text/markdown",
    install_requires=['numpy>=1.17.0', 'cvxpy>=1.1.15', 'ges>=1.0.6']
)
