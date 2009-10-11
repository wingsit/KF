class InputException(Exception):
    def __init__(self):
        print "Check your input"

class ToBeImplemented(Exception):
    def __init__(self, message = ""):
        self.message = message
        print self.message

class DataException(Exception):
    def __init__(self):
        print "Data is bad"
