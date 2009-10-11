"""This file contains a list of statistical function that is useful for time series analysis. Later in the future I am planning to integrate it with statlib"""

class StatException(Exception):
    def __init__(self):
        pass

def mean(lst):
    try: return sum(lst)/len(lst)
    except:
        raise StatException

