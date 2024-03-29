import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

required_package = ['numpy', 'pandas', 'seaborn', 'matplotlib',
'scipy', 'plotly==4.0.0', 'researchpy', 'statsmodels', 'squarify',
'pandas_profiling', 'lxml', 'pivottablejs', 'qgrid']


setuptools.setup(
     name='DataAnalysis',
     version='0.5',
     #scripts=['data_analysis_econometrics'] ,
     author="Thomas Pernet",
     author_email="t.pernetcoudrier@gmail.com",
     description="A simple package to describe the data",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="",
     packages=setuptools.find_packages(),
     install_requires= required_package,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
