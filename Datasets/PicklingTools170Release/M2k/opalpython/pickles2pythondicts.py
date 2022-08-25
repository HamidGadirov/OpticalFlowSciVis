
# Input: A pickle file with multiple things pickled in it:
# Output:  A text file of python pickles in a single file

# Convert some pickles to Python
import cPickle

from pprint import pprint 
import sys
import string

if len(sys.argv) != 3 :
    print "Usage: pickles2pythondicts.py pickleddata.pickle pythontable.py"
    sys.exit(1)

pickle_file = file(sys.argv[1], 'r')
pythontable_file = file(sys.argv[2], 'w')

while 1:
   try :  
     f = cPickle.load(pickle_file)
     if (type(f)==type({}) and f.has_key("CONTENTS")) :
       print "Stripping outer shell with CONTENTS"
       f = f["CONTENTS"]
     pprint(f, pythontable_file)
   except EOFError:
     break

print("All done")







