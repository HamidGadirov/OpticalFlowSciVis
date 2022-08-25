
from pyocser import ocdumps
from pyocser import ocloads
import sys

try :
    import numpy
    #print '** importing NUMPY arrays'
    import numpy_test
except :
    #print " ... but can't import numpy ..."
    numpy = None
    numpy_test = None

try :
    import Numeric
    #print "** importing NUMERIC arrays"
except :
    #print " ... but can't import Numeric ..."
    Numeric = None

# Do we just check dumps, or do we try both?
checking_roundtrip = 0

def check_what_we_can_import () :
    if numpy :
        return 4 # AS_NUMPY
    elif Numeric :
        return 0
    else :
        return 1
    

def eq (o1, o2) : return o1==o2
def numpyeq (o1, o2) : return (o1==o2).all()

def engine (results, mesg, equals=eq) :
    print "---------------", mesg
    for value, result in results :
        trial = ocdumps(value)
        if trial != result :
            print '***ERROR! Expected', value, ' to dumps as\n', repr(result),' not\n', repr(trial)
        else :
            if checking_roundtrip :
                # NOW check that we can deserialize and the same back!
                # print type(trial), trial
                deser = ocloads(trial, 0, check_what_we_can_import())
                if not equals(deser, value) :
                    print '***ERROR! value started as ',value,' and came back as ', deser, " type(original value)=", type(value), " type(return value)=",type(deser)
                else :
                    print '.. ok for', repr(value), ":", repr(deser)
            else :
                print '.. ok for', value, ":", repr(result)


def simpleTests () :
    results = [
        # All potential raw ints
        (1, 'S\x01'),
        (0, 'S\x00'),
        (-1, 's\xff'),
        (127, 'S\x7f'),
        (128, 'S\x80'),
        (129, 'S\x81'),
        (255, 'S\xff'),
        (256, 'I\x00\x01'),
        (-127, 's\x81'),
        (-128, 's\x80'),
        (-129, 'i\x7f\xff'),
        (32767, 'I\xff\x7f'),
        (32768, 'I\x00\x80'),
        (65535, 'I\xff\xff'),
        (-32767, 'i\x01\x80'),
        (-32768, 'i\x00\x80'),
        (-32769, 'l\xff\x7f\xff\xff'),
        (65536,  'L\x00\x00\x01\x00'),
        (65537,  'L\x01\x00\x01\x00'),
        (-2147483647, 'l\x01\x00\x00\x80'),
        (-2147483648, 'l\x00\x00\x00\x80'),
        (4294967295, 'L\xff\xff\xff\xff'),
        
        (None, 'Z'),
        
        # reals and complexes
        (2.0, 'd\x00\x00\x00\x00\x00\x00\x00@'),
        (-100.0, 'd\x00\x00\x00\x00\x00\x00Y\xc0'),
        (2.0-100.0j, 'D\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00Y\xc0'),
        # bool
        (True,  'b\x01'),
        (False, 'b\x00'),
        
        # simple strings
        ("abc", 'a\x03\x00\x00\x00abc'),
        ("",    'a\x00\x00\x00\x00'),
        ("Z234567890"*10, 'ad\x00\x00\x00Z234567890Z234567890Z234567890Z234567890Z234567890Z234567890Z234567890Z234567890Z234567890Z234567890')
        
        ]
           
    engine(results, "simpleTests")


def testIntLongBoundary () :

    if sys.maxint == 2147483647 : # 32-bit machine
        print '---------- 32-bit machine ------------'
        results = [
            ]
    else :
        print '---------- 64-bit machine ------------'
        results = [ 
            # These values are on a 64-bit machine
            (-2147483649, 'x\xff\xff\xff\x7f\xff\xff\xff\xff'),
            (4294967296, 'X\x00\x00\x00\x00\x01\x00\x00\x00'),
            (9223372036854775807, 'X\xff\xff\xff\xff\xff\xff\xff\x7f'),
            (-9223372036854775807, 'x\x01\x00\x00\x00\x00\x00\x00\x80'),
            (-9223372036854775808, 'x\x00\x00\x00\x00\x00\x00\x00\x80'),

            # These tip over to longs
            (-9223372036854775809, 'q\t\x00\x00\x00\xff\xff\xff\xff\xff\xff\xff\x7f\xff'),
            (9223372036854775808, 'q\t\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x80\x00'),
            ]

    engine(results, "testIntAndLongBoundary")
    

def testBigString () :
    a = "1"*(2**32)
    if len(a) != 2**32 :
        print "***ERROR! Expected a very long string!"
    else :
        print '.. ok len for long string'

    ser = ocdumps(a)
    expected_hdr = "A\x00\x00\x00\x00\x01\x00\x00\x00" 
    if ser[:9] != expected_hdr :
        print "***ERROR!  Header on long string is wrong! Saw:", repr(a[:9]), " but expected:", repr(expected_hdr)
    else :
        print '.. ok header on long string'

    if ser[9:] != a :
        print '***ERROR!  Serialized big string wrong'
        sys.exit(1)
    else :
        print '.. ok string serialized correctly'

    if checking_roundtrip :
        # NOW check that we can deserialize and the same back!
        deser = ocloads(ser)
        if deser != a :
            print "***ERROR! didn't get back big string"
        else :
            print '.. ok for big string'
        

def testLongs () :

    results = [
        (18446744073709551616L, 'q\t\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01'),
        (-18446744073709551616L, 'q\x0c\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff')
        ]
    engine(results, "testLongs")


def testArr() :
    l = [1,2,3,4]
    results = [
        ([1, 2.0, "three"], "P\x00\x00\x00\x00\x01\x01 nZ\x03\x00\x00\x00S\x01d\x00\x00\x00\x00\x00\x00\x00@a\x05\x00\x00\x00three"),
        ([], 'P\x00\x00\x00\x00\x01\x01 nZ\x00\x00\x00\x00'),
        ([[],1],'P\x00\x00\x00\x00\x01\x01 nZ\x02\x00\x00\x00P\x01\x00\x00\x00\x01\x01 nZ\x00\x00\x00\x00S\x01' ),
        ([l,l,l], 'P\x00\x00\x00\x00\x01\x01 nZ\x03\x00\x00\x00P\x01\x00\x00\x00\x01\x01 nZ\x04\x00\x00\x00S\x01S\x02S\x03S\x04P\x01\x00\x00\x00P\x01\x00\x00\x00'),
        ]

    engine(results, "testArr")

    # checking shape kept
    ll = [l,l, l]
    ser = ocdumps(ll)
    res = ocloads(ser)
    if res==ll and res[0] is res[0] and res[1] is res[2]:
        print '.. preserved structure for', ll, " and ", res
    else :
        print "***ERROR! Didn't preserve structure for", ll, " and ", res

    if not checking_roundtrip : return
        
    crazy = []
    crazy.append(crazy)
    ser = ocdumps(crazy)
    print '*******************', repr(ser)
    res = ocloads(ser)
    if len(crazy)==1 and len(res)==1 and type(res)==type(crazy) and crazy[0] is crazy :
        print '.. preserved recursive structure for', crazy, " and ", res
    else :
        print "***ERROR! Didn't preserve recursive structure for", crazy, " and ", res
    
def testDict () :
    rec = { }; rec["recursive"] = rec
    results = [
        ( {}, 'P\x00\x00\x00\x00\x01\x01 t\x00\x00\x00\x00'),
        ( {'a':1, 'b':2.0, 'c':'three'},'P\x00\x00\x00\x00\x01\x01 t\x03\x00\x00\x00a\x01\x00\x00\x00aS\x01a\x01\x00\x00\x00ca\x05\x00\x00\x00threea\x01\x00\x00\x00bd\x00\x00\x00\x00\x00\x00\x00@'),
        ( {'a':1, 'b':{}, 'c': {'a':1, 'b':3.0 }}, 'P\x00\x00\x00\x00\x01\x01 t\x03\x00\x00\x00a\x01\x00\x00\x00aS\x01a\x01\x00\x00\x00cP\x01\x00\x00\x00\x01\x01 t\x02\x00\x00\x00a\x01\x00\x00\x00aS\x01a\x01\x00\x00\x00bd\x00\x00\x00\x00\x00\x00\x08@a\x01\x00\x00\x00bP\x02\x00\x00\x00\x01\x01 t\x00\x00\x00\x00'),
        # (rec, 'P\x00\x00\x00\x00\x01\x01 t\x01\x00\x00\x00a\t\x00\x00\x00recursiveP\x00\x00\x00\x00'),
        ]
    engine(results, "testDict")

    if not checking_roundtrip : return
    
    d = { }
    dd = {'a':d, 'b':d, 'c':d }
    ser = ocdumps(dd)
    res = ocloads(ser)
    if res==dd and res['a'] is res['b'] and res['b'] is res['c']:
        print '.. preserved structure for', dd, " and ", res
    else :
        print "***ERROR! Didn't preserve structure for", dd, " and ", res
        
    ser = ocdumps(rec)
    res = ocloads(ser)
    if len(rec)==1 and len(res)==1 and type(rec)==type(res) and res["recursive"] is res :
        print '.. preserved recursive structure for', dd, " and ", res
    else :
        print "***ERROR! Didn't preserve recursive structure for", dd, " and ", res
        

def testTuple () :
    results = [
        ( (1,2.0, "three"),'P\x00\x00\x00\x00\x01\x01 uZ\x03\x00\x00\x00S\x01d\x00\x00\x00\x00\x00\x00\x00@a\x05\x00\x00\x00three'),
        ( (), "P\x00\x00\x00\x00\x01\x01 uZ\x00\x00\x00\x00")
        ]
    engine(results, "testTuple")

    if not checking_roundtrip : return
    
    d = ()
    dd = (d, d, d)
    ser = ocdumps(dd)
    res = ocloads(ser)
    if res==dd and res[0] is res[1] and res[1] is res[2]:
        print '.. preserved structure for', dd, " and ", res
    else :
        print "***ERROR! Didn't preserve structure for", dd, " and ", res

    di = { }
    rec = (di,)
    di["recursive"] = rec
    ser = ocdumps(rec)
    res = ocloads(ser)
    if len(rec)==1 and len(res)==1 and type(rec)==type(res) and res[0]["recursive"] is res :
        print '.. preserved recursive structure for', rec, " and ", res
    else :
        print "***ERROR! Didn't preserve recursive structure for", rec, " and ", res



def testNumPyArray () :

    if numpy is None :
        print " .. can't import numpy .. "
        return

    # Defer evaluation in case your version of Python can't handle numpy
    results = numpy_test.results

    engine(results, "testNumPyAreray", numpyeq)

    if not checking_roundtrip : return

    d= numpy.array([1,2,3], 'i')
    dd = (d, d, d)
    ser = ocdumps(dd)
    res = ocloads(ser)
    if len(dd)==len(res) and (dd[0]==res[0]).all() and res[0] is res[1] and res[1] is res[2]:
        print '.. preserved structure for', dd, " and ", res
    else :
        print "***ERROR! Didn't preserve structure for", dd, " and ", res
    
def testNumericArray () :

    if Numeric is None :
        print " .. can't import Numeric .. "
        return
    
    results = [
        (Numeric.array([1,2,3], 'i'), 'P\x00\x00\x00\x00\x01\x01 nl\x03\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00'),
        (Numeric.array([1,2,3], 'b'), 'P\x00\x00\x00\x00\x01\x01 nS\x03\x00\x00\x00\x01\x02\x03'),
        (Numeric.array([1,2,3], 's'), 'P\x00\x00\x00\x00\x01\x01 ni\x03\x00\x00\x00\x01\x00\x02\x00\x03\x00'),
        ]

    engine(results, "testNumericArray")


def testVERYLARGENumPyArray () :
    if numpy is None :
        print " .. can't import numpy .. "
        return
    
    dd = numpy_test.big_arr()
    print '.. ok dd is big:', len(dd)
    ser = ocdumps(dd)
    print '.. ok Was able to serialize a very big array: len(ser)=', len(ser)
    res = ocloads(ser)
    print '.. ok Was able to deserialize very big array: len(res)=', len(res) 
    if len(dd)==len(res) and (dd[0]==res[0]).all() :
        print '.. worked for big array'
    else :
        print "***ERROR! Didn't work for big array"


def allTests() :
    simpleTests()
    # testBigString()
    testLongs()
    testIntLongBoundary ()
    testArr()
    testDict()
    testTuple()
    testNumPyArray()
    testNumericArray()
    testVERYLARGENumPyArray ()
    
if __name__ == "__main__" :
    checking_roundtrip = 0
    allTests()
    checking_roundtrip = 1
    allTests()

