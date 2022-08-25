

from distutils.core import setup, Extension
setup(name="pyobjconvert", version="1.0",
      ext_modules=[
    Extension("pyobjconvertmodule",
              ["pyobjconvertmodule.cc", "pyobjconverter.cc"],
              include_dirs=['../C++', '../C++/opencontainers_1_8_5/include']),
    ])

setup(name="ocserialization", version="1.0",
      ext_modules=[
    Extension("pyocsermodule",
              ["pyocsermodule.cc", "pyocser.cc"],
              include_dirs=['../C++', '../C++/opencontainers_1_8_5/include']),
    ])

