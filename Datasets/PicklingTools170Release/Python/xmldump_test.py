
from xmltools import *
from pretty    import *
import sys
import StringIO

try :
  from Numeric import array
  supports_numeric = True
except :
  supports_numeric = False
  
try :
  from numpy import array
  supports_numpy = True
except :
  supports_numpy = False

  try :
    from simplearray import SimpleArray as array
    supports_simplearray = True
  except :
    supports_simplearray = False

# Simple test with no attrs
def grindNoAttrs (xd) :
  strs = [ 
    "None", "1", "1.0", "1L", "(1.0-1.0j)", # Primitive types first
    "{ }", "{ 'html': 1 }",
    "{ 'book': { 'date':'2001', 'time':100, 'chapter':['chap1', 'chap2']}}",
    "{ 'book': { 'date':'2001', 'time':100, 'chapter':[{'p1':{'a':1, 'b':2, 'text':'chapter1'}}, 'chap2']}}",
    "{ 'book': { 'date':'2001', 'time':100, 'chapter':[{'p1':{'a':1, 'b':2, 'text':'chapter1'}}, {'text':[1,2,3]}]}}",
    "{ 'book': { 'date':'2001', 'time':100, 'chapter':[{'p1':{'a':1, 'b':2, 'text':'chapter1'}}, [1,2,3]]}}",
    "[1,2,3]",
    "[{'a':1}]",
    "{ 'empty': None, 'alsoempty': {} }",
    "[ [0,1,2], [2,3,4] ]",
  ]

  for s in strs :
    v = eval(s)
    
    sys.stdout.write("As normal:")
    pretty(v)
    print
    
    print "As XML:"
    xd.XMLDumpKeyValue("top",v);
    print


# Quick test using attrs of XML
def grindAttrs (xd) :

  strs = [ 
    "{ 'book': 1, '__attrs__': { 'date':100, 'time':'May 16th, 2010' } }",
    "{ 'simple': 100, 'complex':{ 'a':1 }, '_plug':'in', '__attrs__': { 'date':100, 'time':'May 16th, 2010' } }",
  ]

  for s in strs :
    v = eval(s)
    
    sys.stdout.write("As normal:")
    pretty(v)
    print
    
    print "As XML:" 
    xd.XMLDumpKeyValue("top",v)
    print
  

# Quick test using attrs of XML
def grindESC (xd) :

  strs = [ 
    "{ '123': 'illegal because starts with number' }",
    "{ 'a\"': 'illegal because of quote in name' }",
    "{ 'a<': 'illegal because has <' }",
    "{ 'a>': 'illegal because has >' }",
    "{ 'a&': 'illegal because has &' }",
    "{ 'a\\'': 'illegal because has \\'' }",
    "{ '': 'empty'}",
    "{ 'ok': 'special chars: < > & \" \\' '}",
    "{ 'ok': ['special chars: < > & \" \\' ', '\x11']}", # Note that special unprintable characters gets passed as is ONLY weirdness is escape chars
    "{ '__attrs__': { 'a':1, 'b':2 }, '__content__': 'something' }", # handle special __content__
    "{ 'book': [ {'__attrs__':{'da':1}, '__content__':'junk' } ] }",
    "{ 'ns:book': { 'ns:a1':5 } }", # ya, we need better namespace support
  ]

  for s in strs:
    v = eval(s)
    
    sys.stdout.write("As normal:")
    pretty(v)
    print
    
    print "As XML:" 
    try :
      xd.XMLDumpKeyValue("root",v)
    except Exception, e :
      sys.stdout.write("Saw Error:")
      print e
      print " ... continuing..."
    
    print

try :
  import Numeric
  type_codes = Numeric.typecodes
except :
  try :
    import numpy
    type_codes = numpy.typecodes
  except :
    type_codes = {'Integer': '1sil', 'UnsignedInteger': 'bwu', 'Float': 'fd', 'Character': 'c', 'Complex': 'FD'}


def xlate_typecodes(typecode) :
  # translate from old-style numeric typecode to numpy
  try :
    import Numeric
    return typecode
  except :
    try :
      from numpy import int8, uint8, int16, uint16, int32, uint32, int64
      global type_codes
      type_codes = { 'Integer': [int8, int16, int32, int64],
                     'UnsignedInteger': [int8, int16, int32, int64] }
      numeric_to_numpy = { '1': int8,
                           'b': uint8,
                           's':int16,
                           'w':uint16,
                           'i':int32,
                           'u':uint32,
                           'l':int64,
                           }
      return numeric_to_numpy[typecode]
    except :
      return typecode
  return typecode

def typecoding(vec) :
  try :
    import numpy
    return vec.dtype
  except :
    return vec.typecode()


def helpMeCore (typecode, xd) :
  t = xlate_typecodes(typecode)
  v = array([0]*10, t)
  for ii in xrange(0,10) :
    r = float(float(ii)*123.456789012345678-500.0)
    th = array([int(0)], t)
    if typecoding(th) in type_codes['Integer']:
      r = int(r)
    if typecoding(th) in type_codes['UnsignedInteger'] :
      r = int(r) #unfiged?
    v[ii] = r;
  
  sys.stdout.write("As normal:")
  pretty(v)
  print
  
  print "As XML:"
  xd.XMLDumpKeyValue("top",v)
  print
        
  return v;


def helpMe (typecode, xd) :
  cc = helpMeCore(typecode, xd)
  v = { }
  v["root"] = cc

  sys.stdout.write("As normal:")
  pretty(v)
  print
  
  print "As XML:"
  xd.XMLDumpKeyValue("up",v)
  print

  return cc


# Plain Old Data (POD)
def grindPOD (xd) :
  helpMe('1', xd)
  helpMe('b', xd)
  helpMe('s', xd)
  helpMe('w', xd)
  v1 = helpMe('i', xd)
  helpMe('u', xd)
  helpMe('l', xd)
  # helpMe('l', xd)  
  helpMe('f', xd)
  v2 = helpMe('d', xd)
  helpMe('F', xd)
  helpMe('D', xd)
  
  t = { }
  t["int_4"] = v1
  t["real_4"] = v2
  vv = { }
  vv["data"] = t
  
  sys.stdout.write("As normal:")
  pretty(vv)
  print 
  
  print "As XML:"
  xd.XMLDumpKeyValue("top",vv)
  print

  # make sure we can put in lists
  a = [ ]
  a.append(v1);
  a.append(v2);
  v = { }
  v["data"] = a

  sys.stdout.write("As normal:")
  pretty(v)
  print
  
  print "As XML:"
  xd.XMLDumpKeyValue("top",v)
  print

  vvv = { 'top':{ 'int_4':[ array([1,2,3], 'i'), array([1.0,2.0,3.0], 'd') ] } }
  
  sys.stdout.write("As normal:")
  pretty(vvv)
  print

  pretty(vvv)
  print "As XML:"
  xd.XMLDumpValue(vvv)
  print

def equals(a,b) :
  if type(a) != type(b) : return false
  if supports_numpy :
    if type(a) == type(numpy.array([], 'i')) :
      if len(a) == len(b) and len(a)==0 : return True
      return a.shape == b.shape and numpy.alltrue(a==b) and a.dtype==b.dtype
    # dict
    elif type(a) == dict :
      if len(a) != len(b) : return false
      for key,val in a.iteritems() :
        if not (key in b and equals(b[key], val) ) :
          return False
      return True

    elif type(a) == list :
      if len(a) != len(b) : return False
      for x in xrange(len(a)) :
        if not equals(a[x], b[x]) :
          return False
      return True
    else :
      return a==b
  else :
    return a==b

def listify (o) :
  if type(o) in [dict, OrderedDict] :
    result = o.__class__()
    for k,v in o.iteritems() :
      result[k] = listify(v)
    return result
  elif type(o) in [tuple, list] :
    for x in xrange(0, len(o)) :
      o[x] = listify(o[x])
    return o
  elif type(o) in array_types :
    result = o.tolist()
    return result
  else :
    return o

# import xmlloader
def req (xd, arr_disp) :

  if arr_disp==ARRAYDISPOSITION_AS_NUMERIC :
    from Numeric import array
  elif arr_disp==ARRAYDISPOSITION_AS_NUMPY :
    from numpy import array
  else :
    from simplearray import SimpleArray as array

  strs = [ 

    "[]",
    "[1]",
    "[1,2]",
    "[ [] ] ",
    "[ [1] ] ",
    "[ [1,2] ] ",
    "[ {} ] ",
    "[ {'a':1} ] ",
    "[ {'a':1, 'b':2} ] ",
    "[ {'a':1, 'b':2}, {} ] ",
    "[ {'a':1, 'b':2}, {'a':1, 'b':2} ] ",



    "{ 'top': { 'duh' : [ { 'ds': 1, 'tcai': { 'stuff': 0 }}, {'ds': 2, 'tcai': { 'stuff': 666 }} ] } }",

    "[ { 'ds': 1, 'tcai': { 'stuff': 0 }}, {'ds': 2, 'tcai': { 'stuff': 666 } } ]",

    "[ { 'ds': [ [ { 'bd':1, 'bo':2 }, { 'bd':3, 'bo':4 } ] ], 'tcai': { 'stuff': 0 }}, {'ds': [ [ { 'bd':111, 'bo':2222 }, { 'bd':333, 'bo':444 } ] ], 'tcai': { 'stuff': 0 } } ]",

    "{ 'empty': [] }",
    "{ 'empty': [1] }",
    "{ 'empty': [1,2] }",
    "{ 'empty': [1,2, {}] }",
    "{ 'empty': [1,2, {'a':1}] }",
    "{ 'empty': [1,2, {'a':1, 'b':2}] }",
    "{ 'empty': [{}] }",
    "{ 'empty': [{}, 1] }",
    "{ 'empty': [{}, 1, 2] }",

    "{ 'empty': [{'a':1}] }",
    "{ 'empty': [{'a':1}, 1] }",
    "{ 'empty': [{'a':1}, 1, 2] }",


    "{ 'empty': [{'a':1, 'b':2}] }",
    "{ 'empty': {} }",
    "{ 'empty': [ 1 ] }",

    "{ 'empty': [ [  ] ] }",
    "{ 'empty': [ [1 ] ] }",

    "{ 'empty': [ [{}] ] }",
    "{ 'empty': [ [{'a':1}] ] }",
    "{ 'empty': [ [{'a':1, 'b':2}] ] }",

    "{ 'empty': [ [[]] ] }",
    "{ 'empty': [ [[{}]] ] }",
    "{ 'empty': [ [[{'a':1}]] ] }",
    "{ 'empty': [ [[{'a':1, 'b':2}]] ] }",

    "{ 'empty': [ [{},1] ] }",
    "{ 'empty': [ [{'a':1},1] ] }",
    "{ 'empty': [ [{'a':1, 'b':2},1] ] }",

    "{ 'empty': [ [{},1,2] ] }",
    "{ 'empty': [ [{'a':1},1,2] ] }",
    "{ 'empty': [ [{'a':1, 'b':2},1,2] ] }",

    "{ 'empty': [ [1,2] ] }",
    "{ 'empty': [ [1,2, {}] ] }",
    "{ 'empty': [ [1,2, {'a':1}] ] }",
    "{ 'empty': [ [1,2, {'a':1, 'b':2}] ] }",

    "{ 'empty': [ [{'a':1},2, {'a':1, 'b':2}] ] }",
    "{ 'empty': [ [{'a':1},2, {'a':1, 'b':2}] ] }",

    " [ [ { 'ds': [ [ { 'ts':1, 'td':2 }, { 'ts':4, 'td':5 } ] ], 'tcai': {'hey':1} } ] ]",

    "[ { 'ds': [ [ {'filename':'something', 'timedelta':1, 'empty':[], 'carl':[1] }, {'filename':'else', 'timedelta':2}] ], 'ot': {'a':1, 'b':2 } } ]",

	       "{ 'a':1, 'SOMETHING': "
"[ { 'ds': [ [ {'filename':'something', 'timedelta':1}, {'filename':'else', 'timedelta':2}], [ {'filename':'something', 'timedelta':1}, {'filename':'else', 'timedelta':2}]], 'ot': {'a':1, 'b':2 } } ]"
    " } ",
    "{ 'carl':1, 'empty':{} }",
    "{ 'carl':1, 'empty':[] }",


    "{ 'empty': array([], 'i') }",
    "{ 'empty': [array([], 'i')]}",
    "{ 'empty': [array([1], 'i')] }",
    "{ 'empty': [array([1,2,3], 'i')] }",
       
    "{ 'empty': [array([], 'i'),1]}",
    "{ 'empty': [array([1], 'i'),1] }",
    "{ 'empty': [array([1,2,3], 'i'),1] }",

    "{ 'empty': [array([], 'i'),1,2]}",
    "{ 'empty': [array([1], 'i'),1,2] }",
    "{ 'empty': [array([1,2,3], 'i'),1,2] }",

    
    "{ 'empty': [1, array([], 'i')] }",
    "{ 'empty': [1, array([1], 'i')] }",
    "{ 'empty': [1, array([1,2], 'i')] }",

    "{ 'empty': [1,2] }",
    "{ 'empty': [1,2, array([], 'i')] }",
    "{ 'empty': [1,2, array([1], 'i')] }",
    "{ 'empty': [1,2, array([1,2], 'i')] }",

    "{ 'empty': [1,2, {'a': array([], 'i')}] }",
    "{ 'empty': [1,2, {'a': array([1], 'i')}] }",
    "{ 'empty': [1,2, {'a': array([1,2,3], 'i')}] }",
    "{ 'empty': [1,2, {'a':array([],'i'), 'b':2}] }",
    "{ 'empty': [1,2, {'a':array([1],'i'), 'b':2}] }",
    "{ 'empty': [1,2, {'a':array([1,2,3],'i'), 'b':2}] }",
    "{ 'empty': [{'a':array([], 'i')}] }",
    "{ 'empty': [{'a':array([1], 'i')}] }",
    "{ 'empty': [{'a':array([1,2,3], 'i')}] }",
    "{ 'empty': [{'a':array([], 'i')},1] }",
    "{ 'empty': [{'a':array([1], 'i')},1] }",
    "{ 'empty': [{'a':array([1,2,3], 'i')},1] }",
    "{ 'empty': [{'a':array([], 'i')},1,2] }",
    "{ 'empty': [{'a':array([1], 'i')},1,2] }",
    "{ 'empty': [{'a':array([1,2,3], 'i')},1,2] }",


    "{ 'empty': [ [  ] ] }",
    "{ 'empty': [ [ array([],'i') ] ] }",
    "{ 'empty': [ [ array([1],'i') ] ] }",
    "{ 'empty': [ [ array([1,2,3],'i') ] ] }",

    "{ 'empty': [ [{}] ] }",
    "{ 'empty': [ [{'a':1}] ] }",
    "{ 'empty': [ [{'a':1, 'b':2}] ] }",

    "{ 'empty': [ [[array([],'i')]] ] }",
    "{ 'empty': [ [[array([1],'i')]] ] }",
    "{ 'empty': [ [[array([1,2,3],'i')]] ] }",
    "{ 'empty': [ [[array([1,2,3],'i')]] ] }",

    "{ 'data': array([(1+2j), (3+4j)], 'F') }",

  ]

  for s in strs :
    v = eval(s)

    print "-----------As normal:"
    pretty(v)
    print
    
    print "As XML:" 
    xd.XMLDumpKeyValue("top",v);
    print


    os = StringIO.StringIO()
    ds = XMLDumper(os, XML_DUMP_PRETTY| XML_STRICT_HDR, arr_disp)
    ds.XMLDumpKeyValue("top",v)
    
    s = os.getvalue()
    iss = StringIO.StringIO(s)
    result = ReadFromXMLStream(iss, XML_LOAD_DROP_TOP_LEVEL | XML_LOAD_EVAL_CONTENT, arr_disp)
    pretty(result)

    if arr_disp==ARRAYDISPOSITION_AS_LIST :
      other_v = listify(v)
    else :
      other_v = v
    # compare
    if supports_numeric :
      test = (result!=other_v)
    elif supports_numpy :
      try :
        test = not equals(result, other_v)
      except Exception,e:
        print e
        print "********SHOULD BE SAME!!", repr(result), repr(other_v), repr(result)==repr(other_v), type(result), type(other_v), arr_disp, pretty(v)
        test = not ((result.shape() == other_v.shape()) and (numpy.alltrue(a,b)))
        
    if test :
      if (repr(result)!=repr(other_v)) :  ### KLUDE
          print "********SHOULD BE SAME!!", repr(result), repr(other_v), repr(result)==repr(other_v), type(result), type(other_v), arr_disp, pretty(v)
          sys.exit(1)
        
    # Read dumping POD data slightly differently: dump pod array as data
    result_pod = None

    oss = StringIO.StringIO()
    dss = XMLDumper(oss, XML_DUMP_PRETTY | XML_STRICT_HDR | XML_DUMP_POD_LIST_AS_XML_LIST, arr_disp)
    dss.XMLDumpKeyValue("top",v)
      
    ss = oss.getvalue()
    sys.stdout.write("POD DATA in XML as lists::"+ss+"\n")
    sys.stdout.flush()

    isss = StringIO.StringIO(ss)  ## BUG!!! Should be ss
    result_pod = ReadFromXMLStream(isss, XML_LOAD_DROP_TOP_LEVEL|XML_LOAD_EVAL_CONTENT, arr_disp)
    # cout << "POD DATA result::" << result_pod.prettyPrint(cout);

    if not supports_numpy :
      test = result_pod != other_v
    else :
      test = not equals(result_pod, other_v)
      
    if test : # (result_pod!=other_v) :
      if (repr(result_pod) != repr(other_v)) :  #### KLUDE around empty Numeric lists
	print "********SHOULD BE SAME!!", result_pod,  other_v 
	sys.exit(1)
      
      
# Simple test with no attrs
def TryOptions (xd) :
  strs = [
    "{ 'a':'' }",
    "{ 'a':'123' }",
    "{ 'a':123 }",
    "{ 'a': ' 123'} ",
    "{ 'a': (1+2j) }",
    "{ 'a': '(1+2j)' }",
  ]

  for s in strs :
    v = eval(s);
    
    sys.stdout.write("As normal:")
    pretty(v)
    print
    
    print "As XML:" 
    xd.XMLDumpKeyValue("top",v);
    print
  


def fullTests (arr_dis) :

    print "**************OPTIONS: (no attrs): all defaults" 
    xd = XMLDumper(sys.stdout)
    grindNoAttrs(xd)

  
    print "**************OPTIONS: (no attrs): XML_DUMP_PRETTY" 
    xd = XMLDumper(sys.stdout, XML_DUMP_PRETTY)
    grindNoAttrs(xd)


    print "**************OPTIONS: (no attrs): XML_DUMP_PRETTY | XML_DUMP_STRINGS_AS_STRINGS"
    xd = XMLDumper(sys.stdout, XML_DUMP_PRETTY | XML_DUMP_STRINGS_AS_STRINGS);
    grindNoAttrs(xd)

  
    print "**************OPTIONS: (no attrs): XML_DUMP_PRETTY and indent of 1"
    xd = XMLDumper(sys.stdout, XML_DUMP_PRETTY, arr_dis, 1)
    grindNoAttrs(xd)


    print "**************OPTIONS (WITH attrs!): XML_DUMP_PRETTY and indent of 1"
    xd = XMLDumper(sys.stdout, XML_DUMP_PRETTY, arr_dis, 1)
    grindAttrs(xd)


    print "**************OPTIONS (WITH attrs!): XML_DUMP_PRETTY | XML_DUMP_SIMPLE_TAGS_AS_ATTRIBUTES and indent of 1" 
    xd = XMLDumper(sys.stdout, XML_DUMP_PRETTY | XML_DUMP_SIMPLE_TAGS_AS_ATTRIBUTES, arr_dis, 1)
    grindAttrs(xd)

  
    print "**************OPTIONS (WITH attrs!): XML_DUMP_PRETTY | XML_DUMP_PREPEND_KEYS_AS_TAGS | XML_DUMP_SIMPLE_TAGS_AS_ATTRIBUTES and indent of 1" 
    xd = XMLDumper(sys.stdout,
                   XML_DUMP_PRETTY | XML_DUMP_PREPEND_KEYS_AS_TAGS |
                   XML_DUMP_SIMPLE_TAGS_AS_ATTRIBUTES, arr_dis, 1)
    grindAttrs(xd)
    

    print "**************OPTIONS (WITH attrs!): XML_DUMP_PRETTY | XML_DUMP_PREPEND_KEYS_AS_TAGS and indent of 1" 
    xd = XMLDumper(sys.stdout, 
                   XML_DUMP_PRETTY | XML_DUMP_PREPEND_KEYS_AS_TAGS, arr_dis, 1)
    grindAttrs(xd)
  

  
    print "**************OPTIONS (escape chars): XML_DUMP_PRETTY | XML_DUMP_PREPEND_KEYS_AS_TAGS | XML_DUMP_SIMPLE_TAGS_AS_ATTRIBUTES and indent of 1 and throw error"

    xd = XMLDumper(sys.stdout,
                   XML_DUMP_PRETTY | XML_DUMP_PREPEND_KEYS_AS_TAGS |
                   XML_DUMP_SIMPLE_TAGS_AS_ATTRIBUTES,
                   arr_dis,
                   1, 
                   XML_PREPEND_CHAR, 
                   XMLDumper.THROW_ON_ERROR)
    grindESC(xd)
  

    print "**************OPTIONS (strict hdr): XML_DUMP_PRETTY | XML_STRICT_HDR and indent of 1 and throw error" 

    xd = XMLDumper(sys.stdout,
                   XML_DUMP_PRETTY | XML_STRICT_HDR,
                   arr_dis,
                   1, 
                   XML_PREPEND_CHAR, 
                   XMLDumper.THROW_ON_ERROR)
    grindESC(xd)

    print "**************OPTIONS (POD data): XML_DUMP_PRETTY" 
    
    xd = XMLDumper(sys.stdout, 
                   XML_DUMP_PRETTY,
                   arr_dis,
                   4, 
                   XML_PREPEND_CHAR, 
                   XMLDumper.THROW_ON_ERROR)
    grindPOD(xd)


    print "**************OPTIONS (POD data): XML_DUMP_PRETTY | XML_DUMP_POD_LIST_AS_XML_LIST"
    
    xd = XMLDumper(sys.stdout, 
                   XML_DUMP_PRETTY | XML_DUMP_POD_LIST_AS_XML_LIST,
                   arr_dis,
                   4, 
                   XML_PREPEND_CHAR, 
                   XMLDumper.THROW_ON_ERROR)
    grindPOD(xd)


  
    print "**************OPTIONS (request data): XML_DUMP_PRETTY " 
    print " ... make sure what goes out comes back in!" 
    xd = XMLDumper(sys.stdout, 
                   XML_DUMP_PRETTY,
                   arr_dis,
                   4, 
                   XML_PREPEND_CHAR, 
                   XMLDumper.THROW_ON_ERROR)
    req(xd, arr_dis)
  
  
    print "**************OPTIONS (try some other options): XML_DUMP_PRETT "
    xd = XMLDumper(sys.stdout, 
                   XML_DUMP_PRETTY,
                   arr_dis,
                   4, 
                   XML_PREPEND_CHAR, 
                   XMLDumper.THROW_ON_ERROR)
    TryOptions(xd)

    print "**************OPTIONS (try some other options): XML_DUMP_PRETTY | XML_DUMP_STRINGS_AS_STRINGS "
    xd = XMLDumper(sys.stdout, 
                   XML_DUMP_PRETTY | XML_DUMP_STRINGS_AS_STRINGS,
                   arr_dis,
                   4, 
                   XML_PREPEND_CHAR, 
                   XMLDumper.THROW_ON_ERROR)
    TryOptions(xd)

  
    print "**************OPTIONS (try some other options): XML_DUMP_PRETTY | XML_DUMP_STRINGS_BEST_GUESS "
    xd = XMLDumper(sys.stdout, 
                   XML_DUMP_PRETTY | XML_DUMP_STRINGS_BEST_GUESS,
                   arr_dis,
                   4, 
                   XML_PREPEND_CHAR, 
                   XMLDumper.THROW_ON_ERROR)
    TryOptions(xd)
  
  
    print "**************OPTIONS (try some other options): XML_DUMP_PRETTY | XML_DUMP_PREFER_EMPTY_STRINGS " 
    xd = XMLDumper(sys.stdout, 
                   XML_DUMP_PRETTY | XML_DUMP_PREFER_EMPTY_STRINGS,
                   arr_dis,
                   4, 
                   XML_PREPEND_CHAR, 
                   XMLDumper.THROW_ON_ERROR)
    xd = TryOptions(xd)



import struct
if __name__ == "__main__" :

    if struct.calcsize("L")==4 :
      print '...On a 32-bit machine, some of the converted arrays will print their values as LONGs, but the rest of the test should work.'

    arr_disp = ARRAYDISPOSITION_AS_NUMERIC_WRAPPER
    try :
      import Numeric
      arr_disp = ARRAYDISPOSITION_AS_NUMERIC
    except :
      pass
    try :
      import numpy
      arr_disp = ARRAYDISPOSITION_AS_NUMPY
    except :
      pass
    
    fullTests(arr_dis=arr_disp)
    fullTests(arr_dis=ARRAYDISPOSITION_AS_LIST)

