#!/usr/bin/env python

"""Make it so we can prettyPrint OpalTables without needing XMPY."""

from pretty import indentOut_

# M2Image lookup table : turns escaped sequence into characters for that seq
M2ImageLookup = { '\n':"\\n", '\t':"\\t",  # Python/C++ same
                  "\v":"\\v", '\b':"\\b", '\r':"\\r", "\f":"\\f", "\a":'\\a', # like Python?
                  '\'':"\\'", '"':"\\\"", "\0":"\\x00" # Python/C++ same
                }
from curses.ascii import isprint

def M2Image (s) :
    '''code from m2k to output a string with m2k understanding of literals'''
    global M2ImageLookup
    # The final result
    result = ""
    ender = ""

    # Start of with a quote
    result += '"'

    # Go through, looking for special sequences and output
    for c in s :
        if c in M2ImageLookup :
            ender = M2ImageLookup[c]
        elif isprint(c) :
            ender = c
        else :
            # Build a hexadecimal escape sequence.  The output is always
            # going to be a \x followed by 2 hex digits
            ender = hex(ord(c))
        result += ender
    # End with a quote
    result += '"'
    return result

def opalPrintKey_ (stream, s) :
    """Output a key from a dictionary as an ASCII OpalTable"""
    if type(s)==type(1) or type(s)==type(1.1):  # int
        print >>stream, '"'+str(s)+'" =',
        return
    
    if len(s)==0 or not(s[0].isalpha() or s[0]=='_') :
        print >>stream, M2Image(s),'=',
        return
    
    # When outputting a ket, make sure it can be read back in
    contains_bad_chars = 0
    for x in xrange(0, len(s)) :
        if not(s[x].isalnum() or s[x]=='_') :
            contains_bad_chars = 1
            break
    if contains_bad_chars :
        print >>stream, M2Image(s),'=',
        return
    else :
        print >>stream, s,'=',
        return


    
def prettyPrintOpalDictHelper_ (d, stream, indent, pretty_print=True, indent_additive=4) :
    """Helper routine to print nested dicts and arrays with more structure"""
    
    # Base case, empty table
    entries = len(d)
    if entries==0 :
        print >>stream, "{ }",
        return

    # Recursive  case
    print >>stream, "{",
    if pretty_print: print >>stream

    # Iterate through, printing each element
    ii=0
    keys = d.keys()
    keys.sort()
    for key in keys :  # Sorted order on keys
        if pretty_print : indentOut_(stream, indent+indent_additive)
        opalPrintKey_(stream, key)
        # print >>stream, repr(key)+"=",
        value = d[key]
        specialStreamOpal_(value,stream, indent, pretty_print, indent_additive)
        if entries>1 and ii!=entries-1 :
            stream.write(",")
        if pretty_print: print >>stream
        ii += 1
        
    if pretty_print : indentOut_(stream, indent)        
    print >>stream, "}",


def prettyPrintOpalListHelper_ (l, stream, indent, pretty_print=True, indent_additive=4) :
    """Helper routine to print nested lists and arrays with more structure"""
    
    # Base case, empty table
    entries = len(l)
    if entries==0 :
        print >>stream, "{ }",
        return
    
    # Recursive case
    print >>stream, "{",
    if pretty_print: print >>stream

    # Iterate through, printing each element
    for ii in xrange(0,entries) :
        if pretty_print : indentOut_(stream, indent+indent_additive)
        opalPrintKey_(stream, ii)
        specialStreamOpal_(l[ii],stream, indent, pretty_print, indent_additive)
        if entries>1 and ii!=entries-1 :
            stream.write(",")
        if pretty_print: print >>stream

    if pretty_print : indentOut_(stream, indent); 
    print >>stream, "}",


def prettyPrintOpalStringHelper_ (s, stream, indent, pretty_print=True, indent_additive=4):
    """Helper routine to print strings"""
    print >>stream, M2Image(s),

# List of special pretty Print methods
OutputMethodOpal = { type("string"):prettyPrintOpalStringHelper_,
                     type( { } )   :prettyPrintOpalDictHelper_,
                     type( [] )    :prettyPrintOpalListHelper_
                   }

# Do we have booleans?
HaveBools = 0
try :
    HaveBools = True
except :
    # Okay, no bools
    pass

# Do we have numpy: prefer numpy
Havenumpy = 0
try :
    import numpy
    Havenumpy = 1
except :
    # Ok, no numpy
    pass


# Do we have Numeric
HaveNumeric = 0
try :
    import Numeric  # prefer numpy
    HaveNumeric = 1
except :
    # Okay, no Numeric
    pass

# Do we have Array 
HaveArray = 0
try :
    import array
    HaveArray = 1
except :
    # Okay, no Array
    pass

EncodeOpalTypeTag = { type(1) : "L",   type(1L):"XL", type(1.0):"D",
                      type(1+1j):"CD" }
def simpleOpalToString_ (value, with_tags = 1) :
    """How does M2k print out simple values: differently from Python"""
    res = ""
    if with_tags and type(value) in EncodeOpalTypeTag:
        res = EncodeOpalTypeTag[type(value)]+":"

    if type(value) == type(1+1j) :
        res+='('+repr(value.real)+","+repr(value.imag)+")"
    elif type(value) == type(1L) :
        res += str(value)   #  Don't want L with oputput
    elif value == None :
        res += '"None"'
    elif HaveBools and type(value)==type(True) :
        res += str(int(value))
    elif Havenumpy and type(value)==type(numpy.array([])) :
        res += NumpyArrayAsString_(value)
    elif HaveNumeric and type(value)==type(Numeric.array([])) :
        res += NumericArrayAsString_(value)            
    elif HaveArray and type(value)==type(array.array('l',[])) :
        res += ArrayArrayAsString_(value) 
    else :
        res+=repr(value)
    return res
    

NumericToOpalTag = { '1':"B", 'b':"UB", 's':"I", 'w':"UI",
                     'i':"L", 'u':"UL", 'l':"X", # 'l':"UX", 
                     'f':'F', 'd':'D', 'F':'CF', 'D':'CD',
                     'b':"L" }
def NumericArrayAsString_ (a) :
    """ print numeric array as M2k Vector """
    result = ""
    for x in xrange(0, len(a)) :
        result += simpleOpalToString_(a[x], 0)+","
    return NumericToOpalTag[a.typecode()]+":<"+result[:-1]+">"


NumpyToOpalTag = { "int8": "B", "uint8": "UB", "int16": "I", "uint16": "UI",
                   "int32": "L", "uint32": "UL", "int64": "X", "uint64": "UX",
                   "float32":'F', "float64":"D",
                   "complex64":"CF", "complex128":"CD",
                   "bool": "L" }

def NumpyArrayAsString_ (a) :
    """ print numpy array as M2k Vector """
    result = ""
    for x in xrange(0, len(a)) :
        result += simpleOpalToString_(a[x], 0)+","
    return NumpyToOpalTag[str(a.dtype)]+":<"+result[:-1]+">"


ArrayToOpalTag = { 'c':"B", 'B':"UB", 'h':"I", 'H':"UI",
                   'i':"L", 'I':"UL", 'l':"X", 'L':"UX", 
                   'f':'F', 'd':'D', 'F':'CF', 'D':'CD',
                   'b':'B' }
def ArrayArrayAsString_ (a) :
    """ print array array as M2k Vector """
    result = ""
    for x in xrange(0, len(a)) :
        result += simpleOpalToString_(a[x], 0)+","
    return ArrayToOpalTag[a.typecode]+":<"+result[:-1]+">"


def simplePrettyPrintOpal_ (value, stream, with_tags = True) :
    res = simpleOpalToString_(value, with_tags)
    print >>stream, res,


def specialStreamOpal_ (value, stream, indent, pretty_print, indent_additive) :
    """Choose the proper pretty printer based on type"""
    global OutputMethodOpal
    type_value = type(value)
    if type_value in OutputMethodOpal:  # Special, indent
        output_method = OutputMethodOpal[type_value]
        indent_plus = 0;
        if pretty_print:indent_plus = indent+indent_additive
        output_method(value, stream, indent_plus, pretty_print, indent_additive)
    else :
        simplePrettyPrintOpal_(value, stream)

import sys

def prettyOpal(value, stream=sys.stdout, starting_indent=0, indent_additive=4):
    """Output the given items in such a way as to highlight
    nested structures of dictionaries or Lists, but output as a
    Midas 2k OpalTable.  By default, it prints to sys.stdout, but can
    easily be redirected to any file:
    >>> f = file('goo.txt', 'w')
    >>> pretty({'a':1}, f)
    >>> f.close()
    """
    indentOut_(stream, starting_indent)
    pretty_print = 1
    specialStreamOpal_(value, stream, starting_indent-indent_additive, pretty_print, indent_additive)
    if type(value) == type([]) or type(value)==type({}) : print >>stream


if __name__=="__main__":
    # Test it
    import sys
    a = [1, 'two', 3.5]
    prettyOpal(a)
    prettyOpal(a,sys.stdout,2)
    prettyOpal(a,sys.stdout,2,2)

    prettyOpal(a)
    prettyOpal(a,sys.stdout,1)
    prettyOpal(a,sys.stdout,1,1)

    t = {'a':1, 'b':2}
    prettyOpal(t)
    prettyOpal(t,sys.stdout,2)
    prettyOpal(t,sys.stdout,2,2)

    prettyOpal(t)
    prettyOpal(t,sys.stdout,1)
    prettyOpal(t,sys.stdout,1,1)

    t[1.1] = 17
    t["1a"] = 17
    t["a1"] = 17
    t["hello"] = "123456789\n\t\x8f"
    t["123456789\n\t\x8f"] = "again"
    t["\"hello\""] = "again"
    t["something"] = None
    t["abooltrue"] = True
    t["aboolfalse"] = False
    t["complex"] = 123+456j
    prettyOpal(t)

    ot = {'MACHINE_REP': 'EEEI', 'TRACKS': [{'KEYWORDS': {}, 'UNITS': '', 'AXES': [{'UNITS': 's', 'START': 0.0, 'LENGTH': 4096L, 'NAME': 'Time', 'DELTA': 1.0, 'KEYWORDS': {}}], 'NAME': 'Track 0', 'FORMAT': 'D'}], 'ATTRIBUTE_PACKET': {}, 'TIME_INTERPRETATION': {'AXIS_TYPE': 'CONTINUOUS'}, 'NAME': 'group(,)', 'GRANULARITY': 4096L, 'KEYWORDS': {}, 'FILE_VERSION': 3L, 'TIME': {'UNITS': 's', 'START': 0.0, 'LENGTH': 4096L, 'NAME': 'Time', 'DELTA': 1.0, 'KEYWORDS': {}}}
    prettyOpal(ot)
    
    if HaveNumeric :
        t = { }
        a= Numeric.array([1,2,3])
        #print type(a)
        #print type(a) == type(Numeric.array([1,2,3]))
        t["NUMERIC_int"] = a
        a= Numeric.array([1.5,2.5,5.5])
        t["NUMERIC_float"] = a
        a = Numeric.array([1+1j,2.5+2.5j,3.5])
        t["NUMERIC_complex"] = a
        prettyOpal(t)

    if Havenumpy :
        t = { }
        a= numpy.array([1,2,3])
        #print type(a)
        #print type(a) == type(Numeric.array([1,2,3]))
        t["NUMPY_int"] = a
        a= numpy.array([1.5,2.5,5.5])
        t["NUMPY_float"] = a
        a = numpy.array([1+1j,2.5+2.5j,3.5])
        t["NUMPY_complex"] = a
        prettyOpal(t)

    if HaveArray :
        t = { }
        a1= array.array('l',[1,2,3])
        #print type(a)
        #print type(a) == type(Numeric.array([1,2,3]))
        t["array_int"] = a1
        a2= array.array('d', [1.5,2.5,5.5])
        t["array_float"] = a2
        # Can't really do complex in Array
        #a = array.array('d', [1+1j,2.5+2.5j,3.5])
        #t["NUMERIC_complex"] = a
        a3 = array.array('b', [127, -128])
        t["array_b"] = a3
        prettyOpal(t)
