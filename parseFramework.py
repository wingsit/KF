import csv
from xlwt import *
from string import split
from datetime import date

#Empty cell error
class EmptyCellException(Exception):
    """Attempt to modify empty cell"""
    def __init__(self):
        print "Cannot modify an empty cell"

#Incorrect menu choice error
class IllegalChoiceException(Exception):
    """Chose an incorrect value in a menu"""
    def __init__(self):
        print "Incorrect menu choice.\n"

#Check if cell is empty
def isEmpty(content):
    if len(content) == 0:
        raise EmptyCellException

#month mapping
def monthMapper(mon):
    month = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7, 'Aug':8,\
             'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
    return str(month[mon])

#Determines the numeric digit of a string containing numbers
def getDigit(a):
    #digit dictionary
    digit = dict([(str(x), x) for x in range(0,10)])
    if a in digit:
        return digit[a]
    #added functionality for determining date type
    else:
        return 100

#file choice mapping
def choiceval(a):
    choiceMap = {1:'csvdata', 2:'txtdata'}
    try:
        return choiceMap[a]
    except KeyError:
        raise IllegalChoiceException

#file choice function mapping
def choicefunc(a):
##    choiceMap = {1:readcsv(), 2:readtxt()}            #####PROBLEM!! EXECUTES BOTH FUNCTIONS! MIGHT AN IF STATEMENT BE BETTER?
##    choiceMap[a]
    if key == 1:
        readcsv()
    else:
        readtxt()

#Determines the format of the date, determined by excel
#Looks at first character
def getDateType(date):
    date = str(date)
    first = date[:1]
    x = getDigit(first)
    if len(str(x)) == 1:
        return 2
    else:
        return 1

def toDate(mon, yr):
    
    #Parse month
    newMon = monthMapper(mon)

    #Parse day
    newDay = 1

    #Parse year
    if yr < 50:
        yr += 2000
    elif yr <= 99:
        yr += 1900
    newYr = yr

    return date(newYr, int(newMon), newDay)
    
#Definition for Date Type 1: MMM-YY
class dateType1():
    def __init__(self, date):
        self.date = date

    def parseDate(self):
        date = str(self.date)
        mon = date[:3]

        #Parse year
        tens = date[4:5]
        ones = date[5:6]
        yr = getDigit(tens) * 10 + getDigit(ones)

        return toDate(mon, yr)

#Definition for Date Type 2: YY-MMM
class dateType2():
    def __init__(self, date):
        self.date = date

    def parseDate(self):
        date = str(self.date)
        list = split(date, '-')
        mon = list[1]

        #Parse year
        year = list[0]
        if len(year) == 1:
            yr = getDigit(year)
        else:
            tens = year[:1]
            ones = date[1:2]
            yr = getDigit(tens) * 10 + getDigit(ones)

        return toDate(mon, yr)

#parse date according to type
def dateparse(date):
    if (getDateType(date) == 1):
        d = dateType1(date)
        return d.parseDate()
    elif (getDateType(date) == 2):
        d = dateType2(date)
        return d.parseDate()
