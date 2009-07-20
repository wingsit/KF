from parseFramework import *
import os
import timeSeriesFrame
import scipy

class ToBeImplemented(Exception):
    def __init__(self):
        print "Not yet implmented"

class FormatConverter:
    def __init__(self, path):
        """Input: string path representing name of the file"""
        
        self.path = path
        self.__data = []          #Intermediate list for conversion/manipulation


    #import csv file to list and modify to format correctly
    def readCSV(self):
        self.__data = list(csv.reader(open(self.path, "rb")))
        
        self.__fix()
        
    #import txt file to list and modify to format correctly    
    def readTXT(self):
        f = open(self.path, 'r')
        for x in f:
            self.__data.append(x.split('\t'))

        #removes newline escape sequences
        for x in self.__data:
            for y in x:
                ind = y.find('\n')
                if ind != -1:
                    new = y[:ind]
                    x.remove(y)
                    x.append(new)

        self.__fix()

    #import xls file  to list and modify to format correctly    
    def readXLS(self):
        raise ToBeImplemented
        
        self.__fix()

    #import sql file to list and modify to format correctly  
    def readSQL(self):
        raise ToBeImplemented
        
        self.__fix()

    #import a timeSeriesFrame object and create a list from it
    def readTSF(self):
        cheader = self.path.cheader
        cheader.insert(0, "")
        self.__data.append(cheader)
        for i in self.path.data:
            row = []
            for j in i.tolist():
                for k in j:
                    row.append(k)
            self.__data.append(row)

        for n, i in enumerate(self.path.rheader):
            self.__data[n + 1].insert(0, i)

        print self.__data
        self.path = "tsfData.tsf"


    #function for formatting data in the list (for all file types)
    def __fix(self):
        self.__data.pop(1)
        self.__data.pop(1)

        for r, x in enumerate(self.__data):
            date = x[0]
            try:
                isEmpty(date)
            except EmptyCellException:      #to handle empty top-left cell, and others if necessary
                continue
            x[0] = dateparse(date)

    """
    data conversion to individual file types
    -->"""
    
    #write the list to a csv file
    def toCSV(self):
        raise ToBeImplemented
    #write the list to a txt document
    def toTXT(self):
        raise ToBeImplemented

    #write the list to an excel document
    def toXLS(self):
        """Input: none
        Output: Excel document in current directory; no written output"""

        w = Workbook()
        ws = w.add_sheet('D & C')

        datefmt = 'M/D/YY'
        style = XFStyle()
        style.num_format_str = datefmt

        for r, x in enumerate(self.__data):
            for c, y in enumerate(x):
                if c == 0:
                    ws.write(r, c, self.__data[r][c], style)
                else:
                    ws.write(r, c, self.__data[r][c])

        w.save(os.path.splitext(self.path)[0] + '.xls')

    #write the list to a sql document
    def toSQL(self):
        raise ToBeImplemented

    #write the list to a timeSeriesFrame object
    def toTSF(self):
        """Input: none
        Output: TimeSeriesFrame object; no written output"""

        c = self.__data.pop(0)[1:]
        self.__data = map(list, zip(*self.__data))
        r = self.__data.pop(0)
        for i in xrange(len(self.__data)):
            self.__data[i] = map(float, self.__data[i])
        self.__data = scipy.transpose(scipy.matrix(self.__data))
        return timeSeriesFrame.TimeSeriesFrame(self.__data, r, c)
