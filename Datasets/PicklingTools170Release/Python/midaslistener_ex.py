#!/usr/bin/env python

# Simple example of how to use a MidasListener: there is some port
# that a MidasYeller is sending UDP packets to, and as soon as we
# start listening, we can pick up what the Yeller says.

from midaslistener import *
import sys
import getopt

if __name__=="__main__" :

  # Parse command line options
  try :
    opts,args=getopt.getopt(sys.argv[1:],[],["ser=","arrdisp="])
    if len(args)!=2 : raise error
  except :
    print "Usage: midaslistener_ex.py [--ser=0|1|2|-2|5] [--arrdisp=0|1|2|4] host port"
    sys.exit(1)

  host = args[0]
  port = int(args[1])
  serialization = SERIALIZE_P0  # Not fast, but backwards compatible
  array_disposition = ARRAYDISPOSITION_AS_LIST   # backwards compat
  
  for opt, val in opts :
    if   opt=="--ser" :     serialization     =  int(val) 
    elif opt=="--arrdisp" : array_disposition =  int(val)
    else : assert False, "unhandled option"

  print "Listening on ", host, port

  a = MidasListener(host, port, 1024, serialization, array_disposition)  
  a.open()

  for x in xrange(0, 1000000) :
     mesg = a.recv(5)
     print mesg

