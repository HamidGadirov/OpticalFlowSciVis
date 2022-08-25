#!/usr/bin/env python

# This is a simple example of how to build a MidasYeller (a server
# that talks to a list of given MidasListeners
#   or M2k UDPClients
# with UDP sockets).
# Creating the MidasYeller only sets it up: when you 'addListener' is
# when you actually add Clients.  Since it's UDP, a send to a client
# may or may not get there, but will probably be very fast.


from midasyeller import *
import sys
import getopt
import time

if __name__=="__main__" :

  # Parse command line options
  try :
    opts,args=getopt.getopt(sys.argv[1:],[],["ser=","arrdisp=", "mainport="])
    if len(args)<2 or len(args)%2==1 : raise error
  except :
    print "Usage: midasyeller_ex.py [--ser=0|1|2|-2] [--arrdisp=0|1|2] [--mainport=number] hostname1 portnumber1 [host2 port2] [host3 port3] ..."
    print " ... the hostname,port pairs are the list of listeners"
    print " ... that are listening to this yeller."
    sys.exit(1)

  serialization = SERIALIZE_P0  # Not fast, but backwards compatible
  array_disposition = ARRAYDISPOSITION_AS_LIST   # backwards compat
  mainport = 9711
  for opt, val in opts :
    if   opt=="--ser" :     serialization     =  int(val)
    elif opt=="--arrdisp" : array_disposition =  int(val)
    elif opt=="--mainport": mainport =  int(val)
    else : assert False, "unhandled option"

  my = MidasYeller(mainport, serialization, array_disposition)
  for x in xrange(0, len(args), 2) :
    host = args[x]
    port = int(args[x+1])
    print "Adding:",host, " ", port
    my.addListener(host, port)

  for x in xrange(0,100) :
    data = { 'hello there' : x, 'This is really  kind of a big table':x }
    print 'Just send UDP message',x
    my.send(data)
    time.sleep(2)


