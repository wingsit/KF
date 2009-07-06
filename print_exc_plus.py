import sys, traceback
def print_exc_plus():
    """Print the usual traceback information, followed by a listing of 
    all the local variables in each frame
    """
    tb = sys.exc_info()[2]
    while tb.tb_next:
        tb = tb.tb_next
    stack = []
    f = tb.tb_frame
    while f:
        stack.append(f)
        f = f.f_back
    stack.reverse()
    traceback.print_exc()
    print "Locals by frame, innermost last"
    for frame in stack:
        print
        print "Frame %s in %s at line %s" % (frame.f_code.co_name, frame.f_code.co_filename, frame.f_lineno)
        for key, value in frame.f_locals.items():
            print "\t%20s = " % key,
            try: print value
            except: print "<ERROR WHILE PRINT VALUE>"

if __name__ == "__main__":
    data = ["1","2",3,"4"]
    def pad4(seq):
        return_value = []
        for thing in seq:
            return_value.append("0"*(4-len(thing)) + thing)
        return return_value
    try:
        print pad4(data)
    except: 

        traceback.print_exc()
        print_exc_plus()

