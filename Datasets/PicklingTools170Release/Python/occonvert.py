#!/usr/bin/env python

"""Utilities to convert Python dictionaries (which have keys of 0,1,2..n)
   to arrays and vice-versa."""


def ConvertAllTabsToArrs(v) :
    """Recursively convert all Tabs (Python Dictionaries) which have the form
    with keys 0..n-1 or "0".."n-1" to Python Lists (Arr).  This returns
    a new deep copy."""
    type_of = type(v)
    if type_of == dict :
        res = { }
        for key,value in v.iteritems() :
            res[key] = ConvertAllTabsToArrs(value)
        return ConvertTabToArr(res,True)
    elif type_of == list :
        res = []
        for x in v :
            res.append(ConvertAllTabsToArrs(x))
        return res
    else :
        return v


def CanBeInt(v):
    """Helper routine which detects if a key is an in integer or can VERY
    EASILY be converted to an integer. ("0", "1", "123"). 
    Returns the integer converted to if legal, otherwise returns None."""
    type_of = type(v)
    if type_of==int : return v
    elif type_of==str and v.isdigit() : return int(v)
    else : return None
    
def ConvertTabToArr(d, recurse=True) :
    """If a table has int keys 0..(n-1) (contiguous), it can be
    converted to a Python List (Arr) easily: If there is a gap, or
    some keys are missing, or the keys are simply non-digit strings,
    then this returns the table as is.  If the keys are '0'..n-1 (as
    strings), then this also tries to handle that.  You can either do
    a shallow (top-level only) conversion [recurse=False] or deep-copy
    [recurse=True].   Either way, this always returns a new value."""
    n = len(d)         # Number of keys
    there = [True]*n   # Remember which keys seen
    values = [None]*n  # Remember the values seen
    tab_still_convertible = True
    for (key, value) in d.iteritems() :
        if recurse :
            value = ConvertAllTabsToArrs(value)
        if tab_still_convertible :
            int_key = CanBeInt(key)
            # print 'int_key', int_key, 'type',type(int_key), n
            if int_key!=None and int_key<n and int_key>=0 and there[int_key]:
                there[int_key] = False
                values[int_key]= value
            else :
                tab_still_convertible = False
                        
    # Make sure ALL entries got filled
    for x in there :
        if x : return d   # Any Trues means still never seen

    # Assertion exactly n keys, and all seen
    return values

def ConvertAllArrsToTabs(v, keys_as_strings=False) :
    """Convert, recursively, all Arrs (Python Lists) to Tabs (Python
    dictionaries).  Returns the new result, basically a deep copy.
    The keys are either 0..(n-1) or ("0".."n-1" if keys_as_strings set)"""
    type_of = type(v)
    if type_of == dict :
        res = { }
        for key,value in v.iteritems() :
            res[key] = ConvertAllArrsToTabs(value, keys_as_strings)
        return res
    elif type_of == list :
        res = []
        for x in v :
            res.append(ConvertAllArrsToTabs(x, keys_as_strings))
        return ConvertArrToTab(res, True, keys_as_strings)
    else :
        return v

def ConvertArrToTab (a, recurse=True, keys_as_strings=False) :
    """ Convert a single Python List (Arr) to a Python Dictionary
    (Tab).  The keys of the dictionary become 0..n-1 (where n is the
    length of the list) [or '0'..'n-1' is keys_as_strings set].  You
    can either do a shallow (top-level only) conversion
    [recurse=False] or deep-copy [recurse=True]."""
    res = { }
    int_key = 0 
    for x in a :
        if keys_as_strings :
            key = str(int_key)
        else :
            key = int_key
        if recurse :
            res[key] = ConvertAllArrsToTabs(x)
        else :
            res[key] = x
        int_key += 1
    return res


def TabConvertTest(t) :
    print 'Before:', t
    ff = ConvertAllTabsToArrs(t)
    print 'After:',t, ff

def ArrConvertTest(a, keys_as_strings) :
    print 'Before:', a
    ff = ConvertAllArrsToTabs(a, keys_as_strings)
    print 'After:',a, ff
    
if __name__=="__main__" :
    # Testing Harness
    t = {'a':1, 'b':2}
    TabConvertTest(t)
    t = {0:'a'}
    TabConvertTest(t)
    t = {0:'a', 1:'b', 2:'c'}
    TabConvertTest(t)
    t = {0:'a', 3:'b', 2:'c'}
    TabConvertTest(t)
    t = 17
    TabConvertTest(t)
    t = { "0":'a' }
    TabConvertTest(t)
    t = { "0":'a', "1":'b'}
    TabConvertTest(t)
    t = { "0":'a', "1":'b', '2':'c'}
    TabConvertTest(t)
    t = { "0":'a', "1":'b', '2':'c', 3:'hello'}
    TabConvertTest(t)
    t = { "0":'a', "1":'b', '2':'c', 3:[1,2, {0:1}]}
    TabConvertTest(t)
    t = { "0":'a', "1":'b', '2':'c', 3:[1,2, {'a':1}]}
    TabConvertTest(t)
    t = { "6":'a', "1":'b', '2':'c', 3:[1,2, {'a':1}]}
    TabConvertTest(t)
    t = { "6":'a', "1":'b', '2':'c', 3:[1,2, {'0':1}]}
    TabConvertTest(t)
    t = { "0":'a', "1":'b', 2:[1,2, {'0':1}]}
    TabConvertTest(t)
    t = { "0":'a', "1":'b', "2.2":'j'}
    TabConvertTest(t)
    t = { "0":'a', "1":'b', "2":'j'}
    TabConvertTest(t)

    a = []
    ArrConvertTest(a, True)
    a = []
    ArrConvertTest(a, False)
    
    a = [1,2,3]
    ArrConvertTest(a, True)
    a = [1,2,3]
    ArrConvertTest(a, False)

    a = {'a':[1,2,3], 'b':{'next':1}}
    ArrConvertTest(a, True)
    a = {'a':[1,2,3], 'b':{'next':1}}
    ArrConvertTest(a, False)

    a = {'a':[1,2,3], 'b':{'next':1}}
    ArrConvertTest(a, True)
    a = {'a':[1,2,3], 'b':{'next':1}}
    ArrConvertTest(a, False)


