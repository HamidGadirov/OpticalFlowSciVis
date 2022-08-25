#!/bin/env python

""" Tool demonstrating how to convert from Python Dictionaries to XML"""

from xmltools import *
try :
    from cxmltools import *    # Use CXMLtools if possible
except :
    pass

import sys

if __name__ == "__main__" :

    input_stdin = False
    output_stdout = False
    if (len(sys.argv)==1) :
        # use as filter: stdin and stdout
        input_stdin = True
        output_stdout = True
    elif (len(sys.argv)==2) :
        input_stdin = False
        output_stdout = True
    elif (len(sys.argv)!=3) :
        print >> sys.stderr, "usage:" , sys.argv[0] , "[[input.pythondict] output.xml]"
        print >> sys.stderr, "         With no options, this reads stdin and output to stdout" 
        sys.exit(1)

    if (input_stdin) :
        f = sys.stdin
    else :
        f = file(sys.argv[1], 'r')
        
    arr_dis = ARRAYDISPOSITION_AS_NUMERIC_WRAPPER  
    if arr_dis == ARRAYDISPOSITION_AS_NUMPY :
        # NUMPY needs dtype, float64, etc. for the eval (below) to work
        from numpy import *
        
    v = eval(f.read())


    xml_options =  XML_DUMP_PRETTY | XML_STRICT_HDR  | XML_DUMP_STRINGS_BEST_GUESS # Guess as to whether we need quotes
    if (output_stdout) :
        WriteToXMLStream(v, sys.stdout, "root", xml_options, arr_dis)
    else :
        WriteToXMLFile(v, sys.argv[2], "root", xml_options, arr_dis)
