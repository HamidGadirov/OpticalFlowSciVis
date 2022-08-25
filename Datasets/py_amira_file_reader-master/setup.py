from distutils.core import setup, Command
# you can also import from setuptools

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        import sys
        errno = subprocess.call([sys.executable, 'runtests.py'])
        raise SystemExit(errno)


setup(name='py_amira_file_reader',
      packages=['py_amira_file_reader'],
      version='0.0.1',
      cmdclass = {'test': PyTest},
)
