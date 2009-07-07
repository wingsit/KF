"""This file contains all the utility function that I have not had time to group them"""

from print_exc_plus import print_exc_plus 

def twoIterate(iterable, fcn, init = None):
    """This fucntion takes a iterable, fucntion, and optional initial value.
	First argument: iterable
	Second argument: a fucntion which takes two argument
	Optional third argument: initial value
	This function somewhat acts like map() function in Python but instead of applying the fucntion to each element, this fucntion will apply fcn to previous and current element in the iterable. Since the first element in the iterable does not have any element beside itself, therefore an initial value can be specified. Otherwise, it is skipped."""
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
