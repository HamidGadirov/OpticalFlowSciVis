
from conforms import *
from pretty import pretty
import sys


from simplearray import SimpleArray as array
# from numpy import array as array

class dumb_dict(dict) :
    def __init__(self, init) :
        dict.__init__(self, init)
    def __str__(self) :
        return "dumb_dict("+dict.__str__(self)+")"
    def __repr__(self) :
        return "dumb_dict("+dict.__repr__(self)+")"
    
def mesg (expected, preamble,
	  instance, prototype, 
	  structure_match=True, type_match=EXACT_MATCH,
          thrw=False) :

    # strings for header
    sm = "NO"
    if structure_match : sm = "YES"
    
    if (type_match==EXACT_MATCH):
        conform = "EXACT_MATCH"
    elif (type_match==LOOSE_MATCH):
        conform= "LOOSE_MATCH"
    elif (type_match==LOOSE_STRING_MATCH):
        conform = "LOOSE_STRING_MATCH"
    else :
        raise Exception("Illegal conform?"+str(int(type_match)))
  
    # info about test
    print "---------", preamble, "\n----- structure_match:", sm, " type_match:", conform , " " , thrw 
    print "..............instance............"
    pretty(instance, sys.stdout)
    print
    print "..............prototype............." 
    pretty(prototype, sys.stdout)
    print
    
    # perform conforms
    result = False
    threw = False
    try :
        result = conforms(instance, prototype, structure_match, type_match, thrw)
        if result == None :
            print "!!!!!!!!!!!!!!!!!!!!!!! Should never return None !!!!!!!!!!!"
            sys.exit(1)
            
    except Exception, e :
        threw = True
        print "**MESSAGE CAUGHT:"
        print e

    # result
    if result :
        print "--> result: True"
    else :
        print "--> result: False"
        
    if result!=expected :
        print "!!!!!!!!!!!!!!! NOT EXPECTED !!!!!!!!!!!" 
        sys.exit(1)
  
    if (result==False and thrw==True and threw==False) :
        print "!!!!!!!!!!!!!!! No error message thrown??? !!!!!!!!" 
        sys.exit(1)



# should always be True, under all variations
def easyCheck(instance, prototype, thr) :

    keymatch = [ True, False ]
    trials = [ EXACT_MATCH, LOOSE_MATCH, LOOSE_STRING_MATCH ]
    
    for keymatch_ii in keymatch :
        for trials_jj in trials :
            mesg(True, "easy test", instance, prototype, 
                 keymatch_ii, trials_jj, thr)
            
      

def failCheck(instance, prototype, thr) :

  if type(instance) == type(prototype) : return # really just testing failure,
  # so we just bypass when tag is the same

  keymatch = [ True, False  ]
  trials   = [ EXACT_MATCH, LOOSE_MATCH, LOOSE_STRING_MATCH ]
  
  for keymatch_ii in keymatch :
      for trials_jj in trials :
          
          same_structure = keymatch_ii
          type_match = trials_jj
          expected=False

          # Exceptional cases
          if instance==OrderedDict() and prototype=={} and type_match!=EXACT_MATCH :
              expected=True
          if instance=="" and type_match==LOOSE_STRING_MATCH :
              expected=True
          if type(instance)==long and instance==1L and type(prototype)==int and prototype==1 and (type_match==LOOSE_MATCH or type_match==LOOSE_STRING_MATCH):
              expected = True

          
          mesg(expected, "fail test", instance, prototype, 
               same_structure, type_match, thr)


def testing (thr) :

  # Start simple
  instance = {'a':1, 'b':2.2, 'c':'three'}
  prototype= {'a':0, 'b':0.0, 'c':''}
  mesg(True, "simple test", instance, prototype, 
       True, EXACT_MATCH, thr)
  
  instance = {'a':1, 'b':2.2, 'c':'three'}
  prototype= {'a':0, 'b':0.0}
  mesg(False, "simple test, too many keys", instance, prototype,
       True, EXACT_MATCH, thr)

  instance = {'a':1, 'b':2.2, 'c':'three'}
  prototype= {'a':0, 'b':0.0}
  mesg(True, "simple test, prototype a subset of keys", instance, prototype, 
       False, EXACT_MATCH, thr)

  instance = {'a':1, 'b':2.2, 'c':'three'}
  prototype= {'a':None, 'b':None}
  mesg(True, "simple test, no type match", instance, prototype, 
       False, EXACT_MATCH, thr)
  
  instance = {'a':1, 'b':2.2, 'c':'three'}
  instance["a"] = long(100) # int_u8(100)
  instance["b"] = float(3.141592658) # real_4(3.141592658)
  prototype= {'a':0, 'b':0.0}
  mesg(True, "simple test, no type match", instance, prototype, 
       False, LOOSE_MATCH, thr)

  instance = {'a':1, 'b':2.2, 'c':'three'}
  instance["a"] = long(100) # int_u8(100) int_u8(100)
  instance["b"] = float(3.141592658) # real_4(3.141592658) real_4(3.141592658)
  prototype= {'a':0, 'b':0.0}
  mesg(False, "simple test, no type match", instance, prototype, 
       False, EXACT_MATCH, thr)
  
  instance = {'a':1, 'b':2.2, 'c':'three'}
  instance["a"] = long(100) # int_u8(100)
  instance["b"] = float(3.141592658) # real_4(3.141592658) real_4(3.141592658)
  prototype= {'a':0, 'b':0.0}
  mesg(False, "simple test, no type match", instance, prototype, 
       False, EXACT_MATCH, thr)

  instance = {'a':1, 'b':2.2, 'c':'three'}
  instance["a"] = long(100) # int_u8(100) int_u8(100)int_u8(100)
  instance["b"] = float(3.141592658) # real_4(3.141592658) real_4(3.141592658)real_4(3.141592658)
  prototype= {'a':0, 'b':0.0, 'c':0}
  mesg(False, "simple test, no type match", instance, prototype, 
       False, LOOSE_MATCH, thr)

  instance = {'a':1, 'b':2.2, 'c':'three'}
  instance["a"] = long(100) # int_u8(100) int_u8(100)int_u8(100)
  instance["b"] = float(3.141592658) # real_4(3.141592658) real_4(3.141592658)real_4(3.141592658)
  prototype= {'a':0, 'b':0.0, 'c':0}
  mesg(True, "simple test, no type match", instance, prototype, 
       False, LOOSE_STRING_MATCH, thr)

  instance = [1,2.2, 'three']
  prototype= [0,0.0, '']
  mesg(True, "arr test, no type match", instance, prototype, 
       True, EXACT_MATCH, thr)

  instance = [1,2.2, 'three']
  prototype= [0,0.0, '']
  mesg(True, "arr test, no type match", instance, prototype, 
       False, EXACT_MATCH, thr)

  instance = [1,2.2, 'three']
  prototype= [0,0.0, '']
  mesg(True, "arr test, no type match", instance, prototype, 
       True, LOOSE_MATCH, thr)

  instance = [1,2.2, 'three']
  prototype= [0,0.0, '']
  mesg(True, "arr test, no type match", instance, prototype, 
       True, LOOSE_STRING_MATCH, thr)

  instance = [1,2.2, 'three']
  prototype= [0,0.0, '']
  mesg(True, "arr test, no type match", instance, prototype, 
       False, LOOSE_MATCH, thr)

  instance = [1,2.2, 'three']
  prototype= [0,0.0, '']
  mesg(True, "arr test, no type match", instance, prototype, 
       True, LOOSE_MATCH, thr)

  instance = [1, 2.2, 'three']
  prototype= [0, 0.0 ]
  mesg(False, "arr test, no type match", instance, prototype, 
       True, EXACT_MATCH, thr)

  instance = [1, 2.2, 'three']
  prototype= [0, 0.0 ]
  mesg(True, "arr test, no type match", instance, prototype, 
       False, EXACT_MATCH, thr)

  instance = [1, 2.2, 'three']
  prototype= [0, 0.0 ]
  mesg(True, "arr test, no type match", instance, prototype, 
       False, LOOSE_MATCH, thr)

  instance = [1, 2.2, 'three']
  instance[0] = long(1) # int_u8(1)
  instance[1] = float(2.333333) # real_4(2.333333)
  prototype= [0, 0.0 ]
  mesg(False, "arr test, no type match", instance, prototype, 
       False, EXACT_MATCH, thr)

  instance = [1, 2.2, 'three']
  instance[0] = long(1) # int_u8(1) int_u8(1)
  instance[1] = float(2.333333) # real_4(2.333333) real_4(2.333333)
  prototype= [0, 0.0 ]
  mesg(True, "arr test, no type match", instance, prototype, 
       False, LOOSE_MATCH, thr)

  instance = [1, 2.2]
  prototype= [0, 0.0, '' ]
  mesg(False, "arr test, no type match", instance, prototype, 
       False, LOOSE_MATCH, thr)

  instance = ['1', '2.2', 'three']
  prototype= [0, 0.0, '' ]
  mesg(True, "arr test, no type match", instance, prototype, 
       False, LOOSE_STRING_MATCH, thr)

    
  easyCheck(1.0, None, thr)
  # easyCheck(1.0f, None, thr)
  #easyCheck(int_1(1), None, thr)
  #easyCheck(int_2(1), None, thr)
  #easyCheck(int_4(1), None, thr)
  #easyCheck(int_8(1), None, thr)
  #easyCheck(int_u1(1), None, thr)
  #easyCheck(int_u2(1), None, thr)
  #easyCheck(int_u4(1), None, thr)
  #easyCheck(int_u8(1), None, thr)
  easyCheck(long(1), None, thr)
  easyCheck(complex(1), None, thr)
  # easyCheck(complex_16(1), None, thr)
  # easyCheck(int_n(1), None, thr)
  # easyCheck(int_un(1), None, thr)
  easyCheck({}, None, thr)
  easyCheck(OrderedDict(), None, thr)
  easyCheck([], None, thr)
  easyCheck((), None, thr)
  ##################################################### easyCheck(Array<int_1>(), None, thr)

  easyCheck(1.0, 0.0, thr)
  # easyCheck(1.0f, 0.0f, thr)
  # easyCheck(int_1(1), int_1(0), thr)
  #easyCheck(int_2(1), int_2(0), thr)
  #easyCheck(int_4(1), int_4(0), thr)
  #easyCheck(int_8(1), int_8(0), thr)
  #easyCheck(int_u1(1), int_u1(0), thr)
  #easyCheck(int_u2(1), int_u2(0), thr)
  #easyCheck(int_u4(1), int_u4(0), thr)
  #easyCheck(int_u8(1), int_u8(0), thr)
  easyCheck(complex(1), complex(0), thr)
  #easyCheck(complex_16(1), complex_16(0), thr)
  #easyCheck(int_n(1), int_n(0), thr)
  easyCheck(long(1), long(0), thr)
  #easyCheck(int_un(1), int_un(0), thr)
  easyCheck({}, {}, thr)
  easyCheck(OrderedDict(), OrderedDict(), thr)
  easyCheck([], [], thr)
  easyCheck((), (), thr)
  ############################easyCheck(Array<int_1>(), Array<int_1>(), thr)
  
  instance = {'a':1, 'b':2.2, 'c':'three', 'd':[1,2.2,3.3],
              'nested':{'a':1, 'b':2.2, 'c':'three', 'd':[1,{},[]]}}
  prototype ={'a':1, 'b':2.2, 'c':'three', 'd':[1,2.2,3.3],
              'nested':{'a':1, 'b':2.2, 'c':'three', 'd':[1,{},[]]}}
  easyCheck(instance, prototype, thr)
  
  fails = ['', {}, OrderedDict(), 1, 1.0, (1+1j)]
  for ii in xrange(len(fails)) :
      failval = fails[ii]
      if (type(failval)!=float) :
          failCheck(1.0, failval, thr)
          # failCheck(1.0f, failval, thr)
          
      if (type(failval)!=int) :
          failCheck(int(1), failval, thr)
          failCheck(long(1), failval, thr)
    
      if (type(failval) != complex) :
          failCheck(complex(1), failval, thr)
          
      failCheck(long(1), failval, thr)
      failCheck({}, failval, thr)
      failCheck(OrderedDict(), failval, thr)
      failCheck([], failval, thr)
      failCheck((), failval, thr)
      ## failCheck(Array<int_1>(), failval, thr)
      failCheck("", failval, thr)


  instance = {'a':1, 'b':2.2, 'c':'three', 'd':[1,2.2,3.3],
              'nested':{'a':1, 'b':2.2, 'c':'three', 'd':[1,{},[]]}
              }
  
  prototype = {'a':1, 'b':2.2, 'c':'three', 'd':[1,2.2],
               'nested':dumb_dict({'a':1, 'b':2.2, 'c':'three', 'd':[1,{},[]]})
               }
  
  mesg(False, "arr test, no type match", instance, prototype, 
       True, EXACT_MATCH, thr)
  mesg(False, "arr test, no type match", instance, prototype, 
       False, EXACT_MATCH, thr)
  mesg(False, "arr test, no type match", instance, prototype, 
       True, LOOSE_MATCH, thr)
  #mesg(True, "arr test, no type match", instance, prototype,
  #       False, LOOSE_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       False, LOOSE_MATCH, True)
  mesg(False, "arr test, no type match", instance, prototype, 
       True, LOOSE_STRING_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       False, LOOSE_STRING_MATCH, thr)


  instance = dumb_dict({'a':1, 'b':2.2, 'c':'three', 'd':(1,2.2,3.3),
               'nested':{'a':1, 'b':2.2, 'c':'three', 'd':[1,{},[]]}
               })
  prototype = {'a':1, 'b':2.2, 'c':'three', 'd':[1,2.2],
               'nested':dumb_dict({'a':1, 'b':None,'c':'three', 'd':[1,{},[]]})
               }

  mesg(False, "arr test, no type match", instance, prototype, 
       True, EXACT_MATCH, thr)
  mesg(False, "arr test, no type match", instance, prototype, 
       False, EXACT_MATCH, thr)
  mesg(False, "arr test, no type match", instance, prototype, 
       True, LOOSE_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       False, LOOSE_MATCH, thr)
  mesg(False, "arr test, no type match", instance, prototype, 
       True, LOOSE_STRING_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       False, LOOSE_STRING_MATCH, thr)

  instance = array([1,2,3], 'i')
  prototype = array([1,2,3], 'i')

  mesg(True, "arr test, no type match", instance, prototype, 
       True, EXACT_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       False, EXACT_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       True, LOOSE_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       False, LOOSE_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       True, LOOSE_STRING_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       False, LOOSE_STRING_MATCH, thr)


  instance = array([1,2,3], 'f')
  prototype = array([1,2,3], 'd')
  mesg(True, "arr test, no type match", instance, prototype, 
       True, EXACT_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       False, EXACT_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       True, LOOSE_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       False, LOOSE_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       True, LOOSE_STRING_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       False, LOOSE_STRING_MATCH, thr)
  ##mesg(False, "arr test, no type match", instance, prototype, 
  ##     True, EXACT_MATCH, thr)
  ##mesg(False, "arr test, no type match", instance, prototype, 
  ##     False, EXACT_MATCH, thr)
  ##mesg(True, "arr test, no type match", instance, prototype, 
  ##     True, LOOSE_MATCH, thr)
  ##mesg(True, "arr test, no type match", instance, prototype, 
  ##     False, LOOSE_MATCH, thr)
  ##mesg(True, "arr test, no type match", instance, prototype, 
  ##     True, LOOSE_STRING_MATCH, thr)
  ##mesg(True, "arr test, no type match", instance, prototype, 
  ##     False, LOOSE_STRING_MATCH, thr)

  
  instance = array([1,2,3], 'f')
  prototype = [1,2,3]
  mesg(False, "arr test, no type match", instance, prototype, 
       True, EXACT_MATCH, thr)
  mesg(False, "arr test, no type match", instance, prototype, 
       False, EXACT_MATCH, thr)
  mesg(False, "arr test, no type match", instance, prototype, 
       True, LOOSE_MATCH, thr)
  mesg(False, "arr test, no type match", instance, prototype, 
       False, LOOSE_MATCH, thr)
  mesg(False, "arr test, no type match", instance, prototype, 
       True, LOOSE_STRING_MATCH, thr)
  mesg(False, "arr test, no type match", instance, prototype, 
       False, LOOSE_STRING_MATCH, thr)

  instance = [1,2,3]
  prototype = (1,2,3)
  mesg(False, "arr test, no type match", instance, prototype, 
       True, EXACT_MATCH, thr)
  mesg(False, "arr test, no type match", instance, prototype, 
       False, EXACT_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       True, LOOSE_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       False, LOOSE_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       True, LOOSE_STRING_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       False, LOOSE_STRING_MATCH, thr)
    
  instance = (1,2,3)
  prototype = [1,2,3]
  mesg(False, "arr test, no type match", instance, prototype, 
       True, EXACT_MATCH, thr)
  mesg(False, "arr test, no type match", instance, prototype, 
       False, EXACT_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       True, LOOSE_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       False, LOOSE_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       True, LOOSE_STRING_MATCH, thr)
  mesg(True, "arr test, no type match", instance, prototype, 
       False, LOOSE_STRING_MATCH, thr)               
  
  return 0

if __name__ == "__main__" :

  testing(False) # make sure still works as expected with "just" return False   
  testing(True)  # make sure throws error when problems with good message

