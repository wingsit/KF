import scipy, numpy, csv, itertools
from random import sample
from datetime import date
import numpy
from utility import twoIterate
DEBUG = 0

## Exception Classes for Dataframe##
class DataframeException(Exception):
    """Base class exception for Dataframe"""
    def __init__(self):
        print "some issue with data frame"

class RowHeaderExcpetion(DataframeException):
    def __init__(self):
        print "Some problem with Row Header"
class ColumnHeaderExcpetion(DataframeException):
    def __init__(self):
        print "Some problem with Column Header"

class DataExcpetion(DataframeException):
    def __init__(self):
        print "Some problem with Data"

class ToBeImplemented(DataframeException):
    def __init__(self):
        print "Not yet implmented"



## Class for dataframe ##
class Dataframe:
    """This is the base frame that holds flat 2 dimensional data"""
    def __init__(self, data = None, columnList = None, rowList = None, rown = None, coln = None):
        self.data = data
        if not rowList: self.rowHeader(rowList)
        if not columnList: self.columnHeader(columnList)
        pass
    def __str__(self):
        for i in self.data[:3]:
            print i
        print "............................"
        for i in self.data[-3:]:
            print i
    def rowHeader(self, headerList):
        try:
            self.rheader = list(headerList)
        except:
            raise RowHeaderException

    def columnHeader(self, headerList):
        if isinstance(headerList, str):
            self.cheader = headerList
        else:
            try:
                self.cheader = list(headerList)
            except:
                raise ColumnHeaderException

    def summary(self):
        raise ToBeImplemented




## Class for TimeSeriesFrame derived from Dataframe ##

class TimeSeriesFrame(Dataframe):
    i = 0
    ci = 0
    def __init__(self, data = None, rowList = None, columnList = None, rown = None, coln = None):
        try:
            self.data = scipy.matrix(data)
        except:
            raise DataException        
        if rowList != None: self.rowHeader(rowList)
        if columnList != None: self.columnHeader(columnList)
        if coln and columnList == None:
            self.columnHeader(map(str, range(len(self.data[0]))))
        
    def __len__(self):
        return len(self.rheader)

    def __getitem__(self, key):
        def getIndex(l, key):
            if key == None:
                return None
            try:
                return l.index(key)
            except:
                if min(l) > key or max(l) < key:
                    raise IndexError
                else:
                    for i in xrange(len(l)):
                        if l[i] > key:
                            return i
        #implement single index
        if isinstance(key, slice):
            print "Slice: ", slice
            if isinstance(key.start, date) or isinstance(key.stop, date):
                key = slice(getIndex(self.rheader,key.start), getIndex(self.rheader,key.stop))
            return TimeSeriesFrame(self.data[key], self.rheader[key], self.cheader)
        elif len(key) > 2:
            raise DataframeException
        else:
            if isinstance(key[0].start, date) or isinstance(key[0].stop, date):
                key = list(key)
                key[0] = slice(getIndex(self.rheader,key[0].start), getIndex(self.rheader,key[0].stop))
                key = tuple(key)
            return TimeSeriesFrame(self.data[key], self.rheader[key[0]], self.cheader[key[1]])
      
    def __str__(self):
        size = self.size()
        if size[0] >6 and size[1] > 6:
            cstring = "\t\t"
            for i in self.cheader[:3]: cstring +=(str(i)+", ")
            cstring +="..., "
            for i in self.cheader[-3:]: cstring +=(str(i)+", ")
            bodystring =cstring+"\n"
            for n,i in enumerate(self.data[:3,:]):
                tempstring = (str(self.rheader[n]) + "\t")
                for j in i[:,:3]:
                    tempstring += (str(j).strip('[').strip(']')+",")
                tempstring += "..., "
                for j in i[:,-3:]: 
                    tempstring += (str(j).strip('[').strip(']'))+","
                bodystring += (tempstring +"\n")
            bodystring += "..., \t \t ..., \n"
            for n,i in enumerate(self.data[-3:]):
                tempstring = (str(self.rheader[-(3-n)])+"\t")
                for j in i[:,:3]:
                    tempstring += (str(j).strip('[').strip(']'))+", "
                tempstring += "..., "
                for j in i[:,-3:]:
                    tempstring += (str(j).strip('[').strip(']'))+", "
                bodystring += (tempstring +"\n")
            return bodystring
        if DEBUG: print "size: ", size
        if size[0] < 6 and size[1] < 6:
            cstring = "\t\t"
            cstring +=(self.cheader)
            bodystring =cstring+"\n"
            tempstring = (str(self.rheader[0]) + "\t")
            tempstring += (str(self.data[0,0]).strip('[').strip(']')+",")
            return bodystring+tempstring+"\n\n"
        if size[1]<6:
            cstring = "\t\t" + self.cheader
            bodystring =cstring+"\n"
            for n,i in enumerate(self.data[:3,:]):
                tempstring = (str(self.rheader[n]) + "\t")
                for j in i[:,:3]:
                    tempstring += (str(j).strip('[').strip(']')+",")
                bodystring += (tempstring +"\n")
            if size[0] >6: bodystring += "..., \t \t ..., \n"
            for n,i in enumerate(self.data[-3:]):
                tempstring = (str(self.rheader[-(3-n)])+"\t")
                for j in i[:,:3]:
                    tempstring += (str(j).strip('[').strip(']'))+", "
                bodystring += (tempstring +"\n")
            return bodystring
        if size[0] < 6 and size[1] > 6:
            cstring = "\t\t"
            for i in self.cheader[:3]: cstring +=(str(i)+", ")
            cstring +="..., "
            for i in self.cheader[-3:]: cstring +=(str(i)+", ")
            bodystring =cstring+"\n"
            tempstring = (str(self.rheader[0]) + "\t")
            for j in self.data[:,:3]:
                tempstring += (str(j).strip('[').strip(']')+",")
            tempstring += "..., "
            for j in self.data[:,-3:]: 
                tempstring += (str(j).strip('[').strip(']'))+","
            bodystring += (tempstring +"\n")
            return bodystring

        
    def rowHeader(self, headerList):
        try:
            if isinstance(headerList, date):
                self.rheader = [headerList]
            elif all(map(lambda x:isinstance(x, date), headerList)):
                self.rheader = list(headerList)
        except:
            raise RowHeaderExcpetion


    def size(self):
        return scipy.shape(self.data)

    def columnIterator(self):
        """ This is a generator to iterate across the columns"""
        while self.ci<self.size()[1]:
            yield self[:,self.ci]
            self.ci += 1
        else:
            self.ci = 0
            raise StopIteration



    def rowIterator(self):
        """This return a iterator of data in each day"""
        while self.i < len(self.rheader):
            yield TimeSeriesFrame(self.data[self.i], self.rheader[self.i], self.cheader)
            self.i+=1
        else:
            self.i = 0
            raise StopIteration

def StylusReader(writer):
    def toDate(element):
        try:
            element = map(int, element.split("/"))
        except:
            print "Element", element
            element = map(int, element.split("/"))

        if element[2] <= 50:
            element[2] += 2000
        elif 51<=element[2]<=99:
            element[2] += 1900
        try:
            return date(element[2], element[0], element[1])
        except:
            print element

    writer = list(writer)
    c = writer.pop(0)[1:]
    writer = map(list, zip(*writer))
    r = map(toDate, writer.pop(0))
    for i in xrange(len(writer)):
        writer[i] = map(float, writer[i])
    data = scipy.transpose(scipy.matrix(writer))
    return TimeSeriesFrame(data, r,c)

def windows(iterable, length=2, overlap = 0):
    it = iter(iterable)
    results = list(itertools.islice(it,length))
    while len(results) == length:
        yield scipy.matrix(results)
        results = results[length - overlap:]
        results.extend(itertools.islice(it, length-overlap))
    if results:
        yield scipy.matrix(results)


        
if __name__ =="__main__":
    stock_data = list(csv.reader(open("dodge_cox.csv", "rb")))
#    lipper_data = list(csv.reader(open("t_lipper_daily.csv", "rb")))    
    stock = StylusReader(stock_data)
#    stock.StylusReader(stock_data)
#    print stock
#    print stock[:,1]
    stocks = list(stock.columnIterator())
    for i in stocks:
        print i
        i.data = scipy.matrix(list(twoIterate(i.data, lambda x,y: (x+y)/x, 0))).T
        print i
