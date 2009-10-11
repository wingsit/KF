#from distutils.core import setup
from setuptools import setup
setup(name='kf',
      version='1.0',
      description= 'test',
      author = 'wingsit',
      author_email = 'wing1127aishi@gmail.com',
      url = 'http://github.com/wingsit/kf',
      py_modules=['Regression', 'TimeSeriesFrame',
                  'utility', 'ConstrainedKalmanSmoother',
                  'KalmanFilter', 'ConstrainedFlexibleLeastSQuares',
                  'ConstrainedKalmanFilter', 'Constrainedlwr',
                  'KalmanSmoother', 'print_exc_plus'],
      )
