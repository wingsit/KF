import scipy, itertools

def windows(iterable, window, overlap):
    if not isinstance(iterable, scipy.matrix):
        iterable = list(iterable)
    i = 0
    diff = window - overlap
    while i <=len(iterable)-window:
        yield iterable[i:i+window]
        i += diff

def fullRollingWindows(iterable, window, overlap):
    if not isinstance(iterable, scipy.matrix):
        iterable = list(iterable)
    i = -overlap
    diff = window - overlap
    while i <len(iterable):
        if i < 0:
            yield iterable[:window]
        elif i + window > len(iterable):
            yield iterable[-window:]
        else: 
            yield iterable[i:i+window]
        i += diff

def headRollingWindows(iterable, window, overlap):
    if not isinstance(iterable, scipy.matrix):
        iterable = list(iterable)
    i = -overlap
    diff = window - overlap
    while i <= len(iterable) - window:
        if i < 0:
            yield iterable[:window]
        else: 
            yield iterable[i:i+window]
        i += diff


def main():
    l = xrange(0,30)
    m = scipy.matrix(xrange(60)).reshape((2,30)).T
#     for i in matrixwindows(m,10,9):
#         print i
#     for i in rollingwindows(l,10,9):
#         print i
#    for i in matrixwindows(m,7,4):
#        print i
    for i in fullRollingWindows(m,10,9):
        print i

if __name__ == "__main__":

    main()
