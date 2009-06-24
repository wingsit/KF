from print_exc_plus import print_exc_plus 

def twoIterate(iterable, fcn, init = None):
    """This fucntion takes a iterable, fucntion, and optional initial value."""
    if init!=None:
        yield init
    n = len(iterable)
    i = 0
    while i < n-1:
        yield fcn(iterable[i], iterable[i+1])
        i+=1
    else:
        raise StopIteration


if __name__ == "__main__":
    l = [1,2,3,4,5,6,7,8,9,10]
    f = lambda x,y : (x+y)/float(x)
    print l
    try:
        print list(twoIterate(l,f)) 
    except:
        print_exc_plus()
