import scipy
import itertools
def windows(iterable, length=2, overlap = 0):
    it = iter(iterable)
    results = list(itertools.islice(it,length))
#    while len(results) == length:
    ind = 1
    while ind:
        yield results
        results = results[length - overlap:]
        results.extend(itertools.islice(it, length-overlap))
        if not results:
            ind = False
    if results:
        yield results

def windows2(date, mat, length, overlap, head = False, tail = False):
    if len(date) != len(mat):
        raise IndexError
    n = range(len(date))
    hindex = 0
    if head:
        tindex = -length
    else:
        tindex = 0
    it = iter(date)
    results = list(itertools.islice(it,length))
    ind = True
    while ind:
        tindex += 1
        if head and hindex < length:
            hindex += 1
            yield results
        elif tail and len(date)>= tindex > len(date) - length:
            print "in elif tail and len(date)> tindex > len(date) - length:"
            yield results
            results = results[length - overlap:]
            results.extend(itertools.islice(it, length-overlap))
        elif tail and len(date)+length> tindex > len(date):
            yield date[-length:]
        elif tail and tindex > len(date)+length:
            ind = False
        elif not tail and tindex > len(date):
            print " elif not tail and tindex > len(date)+length:"
            ind = False
        elif not tail and not head and tindex > len(date):
            ind = False
#    if results:
#s        yield results

def main():
    mat = scipy.matrix(xrange(0,100))
    mat = mat.reshape((5,20)).T
    date = range(0, 20)
#    print mat, date
    for i in windows2(date, mat, 10,9, True,True):
        print i
    print
    for i in windows2(date, mat, 10,9, False,True):
        print i
    print

    for i in windows2(date, mat, 10,9, True,False):
        print i
    print

    for i in windows2(date, mat, 10,9, False,False):
        print i

#    for i in windows(date,10,9):
#        print i
    pass

if __name__ == "__main__":
    main()
