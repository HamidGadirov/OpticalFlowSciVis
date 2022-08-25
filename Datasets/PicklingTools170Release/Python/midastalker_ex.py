# Simple example showing how to use the MidasTalker Python class

from midastalker import * # Make sure you have this on your PYTHONPATH
                          # so the import pickes this up (or copy it to cd)
import sys
import string

if len(sys.argv) != 3 :
    print "usage: python midastalker_ex.py hostname portnumber"
    sys.exit(1)

# Create, but don't actually connect
host = sys.argv[1]
port = string.atoi(sys.argv[2])

mt = MidasTalker(host, port)

mt.open() # connect!

# Continue sending until we get an EOF sign 
res = ''
while res!="EOF":
    
  # Create a message (a table) and send it on over to OpalPythonDaemon
  t = { 'a':1, 'hello':[3.3], "nest":{'cc':17, 'dd': 12} }
  mt.send(t)

  # After sending, let's get a response.  This is a blocking call
  res = mt.recv();

  # "Do something" with the result
  print res


