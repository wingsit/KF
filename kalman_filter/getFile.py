import os
from formatConverter import *
import timeSeriesFrame

extmap = {'.csv':1, '.txt':2, '.xls':3, '.sql':4}
dir = os.curdir
"""dir refers to current directory by default
    recommended to change directory to point to a folder with only files desired to be read"""

def __doConv(file, id):
    """Conversion function"""
    
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
    #commented out if not implemented
##    f.toCSV()
##    f.toTXT()
    f.toXLS()
##    f.toSQL()
    print f.toTSF()

def main():
    """traverse directories and search for applicable file types
        Input: none
        Output: name of file being read
                TimeSeriesFrame outputted (according to timeSeriesFrame.TimeSeriesFrame.__str__()
        Creates: filename.xls of file being read

        Test: used csvDC.csv, txtDC.txt
        Sample Output:
        >>> csvDC.csv
        >>> Cannot modify an empty cell
        >>>                 Dodge & Cox Stock, Cash, Top Value, ..., Mid Growth, Sm Value, Sm Growth, 
        >>> 1986-01-01	 1.0016    0.602858  0.752   ,...,  1.923  0.869  2.284,
        >>> 1986-02-01	-1.9514    0.600846  7.406   ,...,  8.816  7.005  7.376,
        >>> 1986-03-01	 14.8853     0.562892   4.668   ,...,  6.004  5.155  4.541,
        >>> ..., 	 	 ..., 
        >>> 2009-03-01	 8.849   0.0205  8.4844, ...,  9.5268  8.8785  8.9754, 
        >>> 2009-04-01	  1.45468990e+01   1.40000000e-02   8.50790000e+00, ...,  14.2105  15.8674  15.0506, 
        >>> 2009-05-01	 7.4948    0.016014  7.2891  , ...,  5.1685  2.1622  3.8719, 

        >>> txtDC.txt
        >>> Cannot modify an empty cell
        >>>                 Dodge & Cox Stock, Cash, Top Value, ..., Mid Growth, Sm Value, Sm Growth, 
        >>> 1986-01-01	 1.0016    0.602858  0.752   ,...,  1.923  0.869  2.284,
        >>> 1986-02-01	-1.9514    0.600846  7.406   ,...,  8.816  7.005  7.376,
        >>> 1986-03-01	 14.8853     0.562892   4.668   ,...,  6.004  5.155  4.541,
        >>> ..., 	 	 ..., 
        >>> 2009-03-01	 8.849   0.0205  8.4844, ...,  9.5268  8.8785  8.9754, 
        >>> 2009-04-01	  1.45468990e+01   1.40000000e-02   8.50790000e+00, ...,  14.2105  15.8674  15.0506, 
        >>> 2009-05-01	 7.4948    0.016014  7.2891  , ...,  5.1685  2.1622  3.8719, 
        """
    __doConv("csvDC.csv", 1)

if __name__ =="__main__":
    main()

    
