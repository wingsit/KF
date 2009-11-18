build: clibregression.pyx
	cython -f -a clibregression.pyx
	gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.6 -I/usr/include/numpy -o clibregression.so clibregression.c
