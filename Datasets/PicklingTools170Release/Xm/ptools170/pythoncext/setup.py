

from distutils.core import setup, Extension
setup(name="pyobjconvert", version="1.0",
      ext_modules=[
    Extension("pyobjconvertmodule",
              ["pyobjconvertmodule.cc", "pyobjconverter.cc"],
	      include_dirs=['../inc']),
    ])

setup(name="ocserialization", version="1.0",
      ext_modules=[
    Extension("pyocsermodule",
              ["pyocsermodule.cc", "pyocser.cc"],
	      include_dirs=['../inc']),
    ])

