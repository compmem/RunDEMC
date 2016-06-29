

from distutils.core import setup, Extension
from distutils.sysconfig import get_config_var
from distutils.extension import Extension

import os
import sys

# set up extensions
ext_modules = []

# define the setup
setup(name='RunDEMC',
      version='0.1.0',
      maintainer=['Per B. Sederberg'],
      maintainer_email=['psederberg@gmail.com'],
      url=['http://github.com/compmem/RunDEMC'],
      packages=['RunDEMC', 'RunDEMC.tests'],
      ext_modules=ext_modules
      )
