class StatException(Exception):
    def __init__(self):
        pass

def mean(lst):
    try: return sum(lst)/len(lst)
    except:
        raise StatException

