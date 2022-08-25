
from xmltools import *
import sys
from pretty import pretty
# Example showing how to read from an XML file and convert to Python dicts

if __name__=="__main__" :

    if len(sys.argv) !=2 : # read from filename provided
        print >> sys.stderr, sys.argv[0], " filename:  reads XML from a stream and converts to dicts"
        sys.exit(1)

    v = ReadFromXMLFile(sys.argv[1], \
                        XML_STRICT_HDR |          # want strict XML hdr
                        XML_LOAD_DROP_TOP_LEVEL | # XML _needs_ top-level
                        XML_LOAD_EVAL_CONTENT     # want real values,not strings
                        )
    pretty(v)
