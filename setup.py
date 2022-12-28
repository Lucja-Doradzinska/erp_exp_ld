
from setuptools import setup

setup(
   name='erp_exp_ld',
   version='0.1.0',
   author='Lucja Doradzinska',
   author_email='l.doradzinska@nencki.edu.pl',
   packages=['erp_exp_ld'],
   url='https://github.com/Lucja-Doradzinska/erp_exp_ld',
   license='LICENSE.txt',
   description='A custom package to analyze EEG data',
   long_description=open('README.md').read(),
   install_requires=[
       "mne >= 0.24.1",
       "autoreject >= 0.2.2",
   ],
)