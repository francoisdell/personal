from PyInstaller import log as logging
from PyInstaller import compat
from os import listdir
from os.path import join
import os
import sys

mkldir = join(compat.base_prefix, "Library", "bin")
print(mkldir)
binaries = [(join(mkldir, mkl), '') for mkl in listdir(mkldir) if mkl.startswith('mkl_')]
print(binaries)

if mkldir not in sys.path:
    sys.path.append(mkldir)

print('\nPATHSEP')
print(*os.pathsep, sep='\n')
print('\nOS PATH')
print(*os.environ['PATH'].split(';'), sep='\n')
for b in binaries:
    if b not in sys.path:
        sys.path.insert(0, b[0])

    if not b[0] in os.environ['PATH']:
        os.environ['PATH'] = b[0] + os.pathsep + os.environ['PATH']

print('\nSYS PATH')
print(*sys.path, sep='\n')

print('\nOS PATH')
print(*os.environ['PATH'].split(';'), sep='\n')