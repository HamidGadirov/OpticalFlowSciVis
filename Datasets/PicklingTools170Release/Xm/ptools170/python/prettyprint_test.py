
# Shows how you can switch between Numeric and NumPy print outs
# if you have to

import pretty

supports_numeric = False
try :
   import Numeric
   supports_numeric = True
   array = Numeric.array
   types_map = pretty.NumericToOCMap
except :
   pass

supports_numpy = False
try :
   import numpy
   supports_numpy = True
   array = numpy.array
   types_map = pretty.NumPyToOCMap
except :
   pass

a = [1.123456789123456789, -100, 123456789 ]

for option_out in [pretty.LIKE_NUMERIC, pretty.LIKE_NUMPY, pretty.NATURAL] :
   pretty.ArrayOutputOption = option_out
   for typecode in types_map :
      print typecode
      
      b = array(a, typecode)
      print b
      pretty.pretty(b)
      print

