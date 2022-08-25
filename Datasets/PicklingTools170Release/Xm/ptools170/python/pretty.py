#!/usr/bin/env python

"""Similar to to pprint from the pprint module, but tends to expose
nested tables and lists much better for a human-readable format. For
example:

 >>> from pretty import pretty
 >>> a = {'a':1, 'b':[1,2], 'c':{'nest':None} }
 >>> print a
 {'a': 1, 'c': {'nest': None}, 'b': [1, 2]}
 >>> pretty(a)
{
    'a':1,
    'b':[
        1,
        2
    ],
    'c':{
        'nest':None
    }
}

Note that pretty always sorts the keys.  This is essentially how prettyPrint
works in OpenContainers and prettyPrint works in Midas 2k OpalTables.
NOTE!  This was recently updated to remove the excess spaces:  the
prettyPrint in Python and C++ should print exactly the same (i.e.,
you can use diff with them)
"""

# Make it so we can print out nan and inf and eval them too!
# IMPORTANT!  You need these in your environment before you eval
# any tables so that you can get eval to work.  For example:
# >>> a = float('inf')
# >>> print a
# inf
# >>> x = { 'i' : a }
# >>> print x
# {'i': inf}
# >>> eval(repr(x))
# Traceback (most recent call last):
#   File "<stdin>", line 1, in ?
#   File "<string>", line 0, in ?
# NameError: name 'inf' is not defined
# >>>
# >>> from pretty import *          # grabs inf and nan
# >>>
# >>> eval(repr(x))                 # Now it works!!
# {'i': inf}
inf = float('inf')
nan = float('nan')

from pprint import pprint

# What do we support?  Prefer numpy
supports_numpy = False
supports_numeric = False
try :
    import numpy
    supports_numpy = True
except :
    try :
        import Numeric   # only if no numpy
        supports_numeric = True
    except :
        pass


# When given a Numeric or Numpy array, how do you output it?
NATURAL = 1      # like it is
LIKE_NUMPY = 2   # like Numpy :  array([1,2,3], dtype=int32)
LIKE_NUMERIC = 3 # like Numeric  array([1,2,3], 'i')
ArrayOutputOption = NATURAL

# TOP-LEVEL: Map to map Val typecodes to Numeric typecodes
# Map for Numeric types
OCToNumericMap = { 's':'1',
                   'S':'b',
                   'i':'s',
                   'I':'w',
                   'l':'i',
                   'L':'u',
                   'x':'l',
                   'X':'l',
                   'f':'f',
                   'd':'d',
                   'F':'F',
                   'D':'D'
                   }

NumericToOCMap = { '1':'s',
                   'b':'S',
                   's':'i',
                   'w':'I',
                   'i':'l',
                   'u':'L',
                   'l':'x',
                   #'X':'l',
                   'f':'f',
                   'd':'d',
                   'F':'F',
                   'D':'D'
                   }

# Map for NumPy types
if supports_numpy :
    NumPyToOCMap = { numpy.dtype('int8'):'s',
                     numpy.dtype('uint8'):'S',
                     numpy.dtype('int16'):'i',
                     numpy.dtype('uint16'):'I',
                     numpy.dtype('int32'):'l',
                     numpy.dtype('uint32'):'L',
                     numpy.dtype('int64'):'x',
                     numpy.dtype('uint64'):'X',
                     numpy.dtype('float32'):'f',
                     numpy.dtype('float64'):'d',
                     numpy.dtype('complex64'):'F',
                     numpy.dtype('complex128'):'D'
                     }
    
OCToNumPyMap = { 's':'int8',
                 'S':'uint8',
                 'i':'int16',
                 'I':'uint16',
                 'l':'int32',
                 'L':'uint32',
                 'x':'int64',
                 'X':'uint64',
                 'f':'float32',
                 'd':'float64',
                 'F':'complex64',
                 'D':'complex128'
                 }


# Not until 2.7, keep as plain dict then
try :
    from collections import OrderedDict
except :
    OrderedDict = dict


def indentOut_ (stream, indent) :
    """Indent the given number of spaces"""
    if indent == 0 :
        return
    else :
        stream.write(" "*indent)

    
def prettyPrintDictHelper_ (d, stream, indent, pretty_print=True, indent_additive=4) :
    """Helper routine to print nested dicts and arrays with more structure"""
    
    # Base case, empty table
    entries = len(d)
    if entries==0 :
        stream.write("{ }")
        return

    # Recursive case
    stream.write("{")
    if pretty_print: stream.write('\n')

    # Iterate through, printing each element
    ii=0
    keys = d.keys()
    keys.sort()
    for key in keys :  # Sorted order on keys
        if pretty_print : indentOut_(stream, indent+indent_additive)
        stream.write(repr(key)+":")
        value = d[key]
        specialStream_(value, stream, indent, pretty_print, indent_additive)
        if entries>1 and ii!=entries-1 :
            stream.write(",")
        if pretty_print: stream.write('\n')
        ii += 1
        
    if pretty_print : indentOut_(stream, indent)        
    stream.write("}")



# TODO: What should the default of OTab pretty print be?
# o{ 'a': 1, 'b':1 } 
# ['a':1, 'b':2]
# OrderedDict([('a',1), ('b':2)])
# Easiest right now is o{ }, but will revisit
# I also like odict() instead of dict.
OTabEmpty=[ "OrderedDict([])", "o{ }","OrderedDict([])" ]
OTabLeft =[ "OrderedDict([", "o{", "[" ]
OTabRight=[ "])", "}", "]" ]
# OC_DEFAULT_OTAB_REPR = 1
if not "OC_DEFAULT_OTAB_REPR" in dir() :
   OC_DEFAULT_OTAB_REPR  = 1
OTabRepr = OC_DEFAULT_OTAB_REPR;

# To change the printing of OrderedDict
# import pretty
# pretty.OTabRepr = 0

def prettyPrintODictHelper_ (d, stream, indent, pretty_print=True, indent_additive=4) :
    """Helper routine to print nested dicts and arrays with more structure"""
    global OTabRepr
    # Base case, empty table
    entries = len(d)
    if entries==0 :
        stream.write(OTabEmpty[OTabRepr]) # "o{ }"
        return

    # Recursive case
    stream.write(OTabLeft[OTabRepr]) # "o{"
    if pretty_print: stream.write('\n')

    # Iterate through, printing each element
    ii=0
    keys = d.keys()
    for key in keys :  # Insertion order on keys
        if pretty_print : indentOut_(stream, indent+indent_additive)
        if OTabRepr == 0 :
            stream.write("("+repr(key)+", ")
        else :
            stream.write(repr(key)+":")
        value = d[key]
        specialStream_(value, stream, indent, pretty_print, indent_additive)
        if OTabRepr == 0 :
            stream.write(")")
            
        if entries>1 and ii!=entries-1 :
            stream.write(",")
        if pretty_print: stream.write('\n')
        ii += 1
        
    if pretty_print : indentOut_(stream, indent)        
    stream.write(OTabRight[OTabRepr])  # "}"


def prettyPrintListHelper_ (l, stream, indent, pretty_print=True, indent_additive=4) :
    """Helper routine to print nested lists and arrays with more structure"""
    
    # Base case, empty table
    entries = len(l)
    if entries==0 :
        stream.write("[ ]")
        return
    
    # Recursive case
    stream.write("[")
    if pretty_print: stream.write('\n')

    # Iterate through, printing each element
    for ii in xrange(0,entries) :
        if pretty_print : indentOut_(stream, indent+indent_additive)
        specialStream_(l[ii], stream, indent, pretty_print, indent_additive)
        if entries>1 and ii!=entries-1 :
            stream.write(",")
        if pretty_print: stream.write('\n')

    if pretty_print : indentOut_(stream, indent); 
    stream.write("]")



def prettyPrintStringHelper_ (s, stream, indent, pretty_print=True, indent_additive=4):
    """Helper routine to print strings"""
    stream.write(repr(s))

# List of special pretty Print methods
OutputMethod = { str           :prettyPrintStringHelper_,
                 OrderedDict   :prettyPrintODictHelper_,
                 dict          :prettyPrintDictHelper_,
                 list          :prettyPrintListHelper_
               }

def formatHelp_ (format_str, value, strip_all_zeros=False) :
    s = format_str % value
    # All this crap: for complex numbers 500.0+0.0j should be 500+0
    # (notice it strips all zeros for complexes) 
    if strip_all_zeros :
        where_decimal_starts = s.find('.')
        if where_decimal_starts == -1 :
            return s   # all done, no 0s to strip after .
        where_e_starts = s.find('E')
        if where_e_starts == -1 :  # no e
            where_e_starts = len(s)
        dot_to_e = s[where_decimal_starts:where_e_starts].rstrip('0')
        if len(dot_to_e)==1 : # just a .
            dot_to_e = ""
        return s[:where_decimal_starts]+dot_to_e+s[where_e_starts:]
    else :
        if not ('E' in s) and s.endswith('0') and '.' in s:
            s = s.rstrip('0')
            if s[-1]=='.' : s+='0'
    return s

def NumericString_ (typecode, value) :
    """ floats need to print 7 digits of precision, doubles 16"""
    if typecode == 'f'   :
        return formatHelp_("%#.7G", value)
    
    elif typecode == 'd' :
        return formatHelp_("%#.16G", value)
    
    elif typecode == 'F' :
        front = '('+formatHelp_("%#.7G", value.real, strip_all_zeros=True)
        if value.imag==0 :
            front += "+0j)"
        else :
            front += formatHelp_("%+#.7G", value.imag, strip_all_zeros=True)+"j)"
        return front
        
    elif typecode == 'D' :
        front = '('+formatHelp_("%#.16G", value.real, strip_all_zeros=True)
        if value.imag==0 :
            front += "+0j)"
        else :
            front += formatHelp_("%+#.16G", value.imag, strip_all_zeros=True)+"j)"
        return front
    
    else :
        return str(value)
    
def specialStream_ (value, stream, indent, pretty_print, indent_additive) :
    """Choose the proper pretty printer based on type"""
    global OutputMethod
    type_value = type(value)
    if type_value in OutputMethod:  # Special, indent
        output_method = OutputMethod[type_value]
        indent_plus = 0;
        if pretty_print:indent_plus = indent+indent_additive
        output_method(value, stream, indent_plus, pretty_print, indent_additive)

    # Numeric array
    elif supports_numeric and type_value == type(Numeric.array([])) :
        stream.write('array([')
        l = value.tolist()
        typecode = value.typecode()
        for x in xrange(0,len(l)) :
            r = NumericString_(typecode, l[x])
            stream.write(r)
            if x<len(l)-1 : stream.write(",")
        stream.write('], ')
        # How to write the tag? Like Numeric or NumPy
        if ArrayOutputOption == NATURAL or ArrayOutputOption == LIKE_NUMERIC :
            stream.write(repr(value.typecode())+")")
        elif ArrayOutputOption == LIKE_NUMPY :
            code = OCToNumPyMap[NumericToOCMap[value.typecode()]]
            stream.write('dtype='+code+')')
        else :
            raise Exception('Unknown ArrayOutputOption for pretty')

    # NumPy Array
    elif supports_numpy and type_value == type(numpy.array([])) :
        stream.write('array([')
        l = value.tolist()
        # Get proper typecode
        lookup = { numpy.dtype('float32') : 'f',numpy.dtype('float64'): 'd',
                   numpy.dtype('complex64'): 'F',numpy.dtype('complex128'):'D' }
        if value.dtype in lookup :
            typecode = lookup[value.dtype]
        else:
            typecode = None
        for x in xrange(0,len(l)) :
            r = NumericString_(typecode, l[x])
            stream.write(r)
            if x<len(l)-1 : stream.write(",")
        stream.write('], ')
        
        # How to write the tag? Like Numeric or NumPy
        if ArrayOutputOption == NATURAL or ArrayOutputOption == LIKE_NUMPY:
            stream.write('dtype='+str(value.dtype).split("'")[0]+")")
        elif ArrayOutputOption == LIKE_NUMERIC :
            typecode = OCToNumericMap[NumPyToOCMap[value.dtype]]
            stream.write(repr(typecode)+")")
        else :
            raise Exception('Unknown ArrayOutputOption for pretty')
        
    elif type_value in [float, complex] : 
        typecode = { float: 'd', complex: 'D' }
        stream.write(NumericString_(typecode[type_value], value))
    else :
        stream.write(repr(value))

import sys

def pretty (value, stream=sys.stdout, starting_indent=0, indent_additive=4) :
    """Output the given items in such a way as to highlight
    nested structures of Python dictionaries or Lists.  By default,
    it prints to sys.stdout, but can easily be redirected to any file:
    >>> f = file('goo.txt', 'w')
    >>> pretty({'a':1}, f)
    >>> f.close()
    """
    indentOut_(stream, starting_indent)
    pretty_print = 1
    specialStream_(value, stream, starting_indent-indent_additive, pretty_print, indent_additive)
    if type(value) in [list, dict, OrderedDict] :
        stream.write('\n')


if __name__=="__main__":
    # Test it
    import sys
    a = [1, 'two', 3.1]
    pretty(a)
    pretty(a,sys.stdout,2)
    pretty(a,sys.stdout,2,2)

    pretty(a)
    pretty(a,sys.stdout,1)
    pretty(a,sys.stdout,1,1)

    t = {'a':1, 'b':2}
    pretty(t)
    pretty(t,sys.stdout,2)
    pretty(t,sys.stdout,2,2)

    pretty(t)
    pretty(t,sys.stdout,1)
    pretty(t,sys.stdout,1,1)

