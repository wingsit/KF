from distutils.core import setup
import sys

sys.path.append('KF')

setup(name='KF',
      version='0.0.1',
      author='Wing H Sit',
      author_email='wing1127aishi@gmail.com',
      url='http://github.com/wingsit/KF',
      download_url='http://github.com/wingsit/KF',
      description='Fund Tracker ',
      package=['regression'],
      provides=['regression'],
      license='BSD',
      classifiers=['Development Status :: 5 - Production/Stable',
                   'Intended Audience :: Developers',
                   'Natural Language :: English',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 2',
                   'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
                   'License :: OSI Approved :: GNU Affero General Public License v3',
                   'Topic :: Internet',
                   'Topic :: Internet :: WWW/HTTP',
                   'Topic :: Scientific/Engineering :: GIS',
                  ],
     )
