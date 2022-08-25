#!/bin/env python

""" Tool demonstrating how to convert from XML to Python Dictionaries"""

from xmltools import *
try :
    from cxmltools import *   # Use C extension modules if you can
except :
    pass
    
from pretty import pretty
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
        print >> sys.stderr, "usage:" , sys.argv[0] , "[[input.xml] output.pythondictionary]"
        print >> sys.stderr, "         With no options, this reads stdin and output to stdout" 
        sys.exit(1)

    xml_options = XML_STRICT_HDR | XML_LOAD_DROP_TOP_LEVEL | XML_LOAD_EVAL_CONTENT 
    arr_disp = ARRAYDISPOSITION_AS_NUMPY # ERIC_WRAPPER

    if arr_disp == ARRAYDISPOSITION_AS_NUMPY :
        from numpy import *
    if (input_stdin) :
        v = ReadFromXMLStream(sys.stdin, xml_options, arr_disp)
    else :
        v = ReadFromXMLFile(sys.argv[1], xml_options, arr_disp)

    if (output_stdout) :
        f = sys.stdout
    else :
        f = file(sys.argv[2], 'w')
    pretty(v, f)

