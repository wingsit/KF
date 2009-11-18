from setuptools import setup, find_packages
#from distutils.core import setup, find_packages
#import py2exe
setup(name='KF',
      version='0.1.2',
      install_requires = ['scipy', 'cvxopt', 'matplotlib', 'numpy'],
      description = ("Fund performance tracker"),
      keywords = 'kalman filter regression fund performance replication tracker',
      author = "Leon Sit",
      author_email = "wing1127aishi@gmail.com",
      url = "www.github.com/wingsit/KF.git",
      license = 'BSD',
      packages = find_packages(),
      )

