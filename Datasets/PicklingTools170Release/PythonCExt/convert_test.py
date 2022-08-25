
import sys
#sys.path.append('/home/rts/Pickling/PicklingTools141BETA/PythonCExt/build/lib.linux-i686-2.6')
sys.path.append('/home/rts/Pickling/PicklingTools141BETA/PythonCExt/build/lib.linux-x86_64-2.4')
sys.path.append('/home/rts/Pickling/PicklingTools141BETA/Python')
from pyobjconvert import *
from xmlloader import *
from xmldumper import *

print '... basic types ...'
a = 1
res = ConvertToVal(a,1); bb = deepcopy_via_val(a); print bb, bb==a
a = -1
res = ConvertToVal(a,1); bb = deepcopy_via_val(a); print bb, bb==a
a = 1.1
res = ConvertToVal(a,1); bb = deepcopy_via_val(a); print bb, bb==a
a = -1.1
res = ConvertToVal(a,1); bb = deepcopy_via_val(a); print bb, bb==a
a = 1.123456789123456789
res = ConvertToVal(a,1); bb = deepcopy_via_val(a); print bb, bb==a
a = 1.23456789123456789 + 987654321.987654321j
res = ConvertToVal(a,1); bb = deepcopy_via_val(a); print bb, bb==a
a = 1+1j
res = ConvertToVal(a,1); bb = deepcopy_via_val(a); print bb, bb==a
a = ""
res = ConvertToVal(a,1); bb = deepcopy_via_val(a); print bb, bb==a
a = "the quick brown fox jumped over the lazy dogs"
res = ConvertToVal(a,1); bb = deepcopy_via_val(a); print bb, bb==a
a = 247365780263456273465897265289763457862398589276345762893465892635L
res = ConvertToVal(a,1); bb = deepcopy_via_val(a); print bb, bb==a
a = 1L
res = ConvertToVal(a,1); bb = deepcopy_via_val(a); print bb, bb==a
a = None
res = ConvertToVal(a,1); bb = deepcopy_via_val(a); print bb, bb==a
a = {}
res = ConvertToVal(a,1); bb = deepcopy_via_val(a); print bb, bb==a
a = {'a':1}
res = ConvertToVal(a,1); bb = deepcopy_via_val(a); print bb, bb==a
a = {'a':1, 'b':2.2}
res = ConvertToVal(a,1); bb = deepcopy_via_val(a); print bb, bb==a
a = {'a':1, 'b':2.2, 'nest': {'a':1} }
res = ConvertToVal(a,1); bb = deepcopy_via_val(a); print bb, bb==a
a= []
res = ConvertToVal(a,1); bb = deepcopy_via_val(a); print bb, bb==a
a= [1,2,3]
res = ConvertToVal(a,1); bb = deepcopy_via_val(a); print bb, bb==a 

b = []
l = [b,b]
res = ConvertToVal(l, 1)

bb= deepcopy_via_val(l)
print bb
print bb[0] is bb[1], l[0] is l[1]


b = []
c = { 'a':1, 'b':2 }
l = [b,b,c,c]
res = ConvertToVal(l, 1)

bb= deepcopy_via_val(l)
print bb
print bb[0] is bb[1], l[0] is l[1]
print bb[2] is bb[3], l[2] is l[3]

b = []
c = {'a':1, 'b':2 }
l = [b, c]
l[1]['backup'] = b
res = ConvertToVal(l, 1)
bb= deepcopy_via_val(l)
print bb
print b is l[1]['backup']

try :
    from collections import OrderedDict
    print '... collections.OrderedDict AVAILABLE! ... '
    a = OrderedDict()
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMPY);
    print repr(bb), bb==a, type(bb) == type(a)
    

    a = OrderedDict([('a',1), ('b',2.2)])
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMPY);
    print repr(bb), bb==a, type(bb) == type(a)

    a = OrderedDict([('a',1), ('b',2.2)])
    b = OrderedDict([('a',a), ('b',a)])
    res = ConvertToVal(b, 1); bb = deepcopy_via_val(b, ARRAYDISPOSITION_AS_NUMPY);
    print repr(bb), bb==b, type(bb) == type(b)
    print b['a'] is b['b']
    pass
except Exception,e :
    print '... collections.OrderedDict not available ...'
    

try :

    import numpy
    print '... numpy AVAILABLE! ...'
    a = numpy.array([1,2,3], 'int8')
    res = ConvertToVal(a, 1);
    bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMPY);
    print repr(bb), bb==a, bb.dtype == a.dtype
    
    a = numpy.array([1,2,3], 'int16')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMPY); print repr(bb), bb==a, bb.dtype==a.dtype
    a = numpy.array([1,2,3], 'int32')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMPY); print repr(bb), bb==a, bb.dtype==a.dtype
    a = numpy.array([1,2,3], 'int64')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMPY); print repr(bb), bb==a, bb.dtype==a.dtype
    a = numpy.array([1,2,3], 'uint8')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMPY); print repr(bb), bb==a, bb.dtype==a.dtype
    a = numpy.array([1,2,3], 'uint16')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMPY); print repr(bb), bb==a, bb.dtype==a.dtype
    a = numpy.array([1,2,3], 'uint32')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMPY); print repr(bb), bb==a, bb.dtype==a.dtype
    a = numpy.array([1,2,3], 'uint64')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMPY); print repr(bb), bb==a, bb.dtype==a.dtype
    a = numpy.array([1,2,3], 'float32')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMPY); print repr(bb), bb==a, bb.dtype==a.dtype
    a = numpy.array([1,2,3], 'float64')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMPY ); print repr(bb), bb==a, bb.dtype==a.dtype
    a = numpy.array([1,2,3], 'complex64')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMPY ); print repr(bb), bb==a, bb.dtype==a.dtype
    a = numpy.array([1,2,3], 'complex128')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMPY ); print repr(bb), bb==a, bb.dtype==a.dtype

    b = [a,a]
    res = ConvertToVal(b, 1); bb = deepcopy_via_val(b, ARRAYDISPOSITION_AS_NUMPY ); print repr(bb)
    print bb[0] is bb[1], b[0] is b[1]

except Exception, e:
    print "... numpy not available ..."


try :

    import Numeric
    print '... Numeric AVAILABLE! ...'
    a = Numeric.array([1,2,3], '1')
    res = ConvertToVal(a, 1);
    bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMERIC);
    print repr(bb), bb==a, bb.typecode() == a.typecode()
    
    a = Numeric.array([1,2,3], 's')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMERIC); print repr(bb), bb==a, bb.typecode()==a.typecode()
    a = Numeric.array([1,2,3], 'i')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMERIC); print repr(bb), bb==a, bb.typecode()==a.typecode()
    a = Numeric.array([1,2,3], 'l')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMERIC); print repr(bb), bb==a, bb.typecode()==a.typecode()
    a = Numeric.array([1,2,3], 'b')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMERIC); print repr(bb), bb==a, bb.typecode()==a.typecode()
    a = Numeric.array([1,2,3], 'w')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMERIC); print repr(bb), bb==a, bb.typecode()==a.typecode()
    a = Numeric.array([1,2,3], 'u')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMERIC); print repr(bb), bb==a, bb.typecode()==a.typecode()
    a = Numeric.array([1,2,3], 'l')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMERIC); print repr(bb), bb==a, bb.typecode()==a.typecode()
    a = Numeric.array([1,2,3], 'f')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMERIC); print repr(bb), bb==a, bb.typecode()==a.typecode()
    a = Numeric.array([1,2,3], 'd')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMERIC ); print repr(bb), bb==a, bb.typecode()==a.typecode()
    a = Numeric.array([1,2,3], 'F')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMERIC ); print repr(bb), bb==a, bb.typecode()==a.typecode()
    a = Numeric.array([1,2,3], 'D')
    res = ConvertToVal(a, 1); bb = deepcopy_via_val(a, ARRAYDISPOSITION_AS_NUMERIC ); print repr(bb), bb==a, bb.typecode()==a.typecode()

    b = [a,a]
    res = ConvertToVal(b, 1); bb = deepcopy_via_val(b, ARRAYDISPOSITION_AS_NUMERIC ); print repr(bb)
    print bb[0] is bb[1], b[0] is b[1]
    

except Exception, e:
    print "... Numeric not available ..."
