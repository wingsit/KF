import scipy, numpy, csv, itertools
from random import sample
from datetime import date
import numpy
from utility import twoIterate
from print_exc_plus import print_exc_plus

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

    def toCSV(self, name = "default.csv"):
        import csv 
        csvReader = csv.writer(open(name, 'w'), dialect='excel')
        for i in self.data.tolist():
            csvReader.writerow(i)
        del csvReader



## Class for TimeSeriesFrame derived from Dataframe ##

class TimeSeriesFrame(Dataframe):
    i = 0                       # Counter for RowIterator
    ci = 0                      # Counter for ColumnIterator
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
        """Return number of time series it has"""
        return len(self.rheader)

    def __getitem__(self, key):
        """Valid Syntax
        stock[:,1],
        stock[:, n:m:r],
        stock[date1:date2],
        stock[date1:date2,:],
        stock[date1:date2, n:m:r]
        stock[date]
        """
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
        if isinstance(key, date): # check stock[date]
            key = getIndex(self.rheader, key)
            return TimeSeriesFrame(self.data[key], self.rheader[key], self.cheader)
        if isinstance(key, slice): # check stock[date1:date2]
            if isinstance(key.start, date) or isinstance(key.stop, date):
                key = slice(getIndex(self.rheader,key.start), getIndex(self.rheader,key.stop))
            return TimeSeriesFrame(self.data[key], self.rheader[key], self.cheader)
        elif len(key) > 2:      # allow two dimensions
            raise DataframeException
        else:
            if isinstance(key[0].start, date) or isinstance(key[0].stop, date): #stock[date1:date2,:], stock[date1:date2, n:m:r]
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
        if size[0] < 6 and size[1] < 4:
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
        """ This is a generator to iterate across different time series"""
        while self.ci<self.size()[1]:
            yield self[:,self.ci]
            self.ci += 1
        else:
            self.ci = 0
            raise StopIteration


    def rowIterator(self):
        """ This is a generator to iterate all the time series by date"""        
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
    #stock.StylusReader(stock_data)
    try: print stock
    except: print "stock"
    try: print stock[:,1]
    except: print "stock[:,1]"
    try: print stock[:, 1:6]
    except: print "stock[:, 1:6]"
    try:
        from datetime import date
        print stock[date(2001,1,1):date(2002,1,1)]
    except: print "stock[date(2001,1,1):date(2002,1,1)]"
    try: print stock[date(2001,1,1):date(2002,1,1),:]
    except: print "stock[date(2001,1,1):date(2002,1,1),:]"
    try: print stock[date(2001,1,1):date(2002,1,1),1:6]
    except: print "stock[date(2001,1,1):date(2002,1,1),1:6]"
    try: print stock[date(2001,1,1)]
    except: 
        print "stock[date(2001,1,1)]"
        print_exc_plus()
    
