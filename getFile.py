import os.path
from FormatConverter import *
import timeSeriesFrame

extmap = {'.csv':1, '.txt':2, '.xls':3, '.sql':4}
dir = "C:\Documents and Settings\MARY\My Documents\Test\Data"

#Conversion function
def doConv(file, id):
    f = FormatConverter(file)       #creates a FormatConverter object for the file

    #reads in file with correct function
    if id == 1:
        f.readCSV()
    elif id == 2:
        f.readTXT()
    elif id == 3:
        f.readXLS()
    elif id == 4:
        f.readSQL()

    #converts to all other file types
##    f.toCSV()
##    f.toTXT()
    f.toXLS()
##    f.toSQL()
    print f.toTSF()



if __name__ =="__main__":
    #traverse directories and search for applicable file types
    def callback( arg, dirname, fnames ):
        for file in fnames:
            ext = os.path.splitext(file)[1]
            if ext in extmap:
                print file
                doConv(file, extmap[ext])

    arglist = []
    os.path.walk(dir,callback,arglist)
    
