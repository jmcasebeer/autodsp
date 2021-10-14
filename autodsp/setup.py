from setuptools import setup

setup(
   name='autodsp',
   version='0.0.1',
   description='Code to reproduce the 2021 WASPAA paper titled AUTO-DSP: LEARNING TO OPTIMIZE ACOUSTIC ECHO CANCELLERS.',
   author='Jonah Casebeer, Nicholas J. Bryan, Paris Smaragdis',
   author_email='jonahmc2@illinois.edu',
   url='https://github.com/jmcasebeer/autodsp',
   packages=['autodsp'],
   license='University of Illinois Open Source License',
   install_requires=[
      'matplotlib==3.4.3',
      'numpy==1.21.2',
      'pandas==1.3.3',
      'scipy==1.7.1',
      'tqdm==4.62.3',
      'wandb==0.12.4',
   ]
)
