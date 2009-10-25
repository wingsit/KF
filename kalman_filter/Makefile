\libregression: libregression.c
	cython -a clibregression.pyx
	clang  -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.6 -o clibregression.so clibregression.c
