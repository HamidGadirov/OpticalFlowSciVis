#!/usr/bin/env python

# A more complicated example showing how to talk to OpalPythonDaemon
# and MidasServers, but still 
# (a) recover from errors
# (b) use timeouts on receieves


from midastalker import * # Make sure you have this on your PYTHONPATH
                          # so the import picks this up (or copy it to cd)
import sys
import string
import time
import getopt

# Main loop where you send and receive messages to the MidasTalker (mt)
def sendloop (mt, data) :

  while 1 :
      
    # Send data to permute
    print 'Sending request:', data
    mt.send(data)

    # See if we can receive, wait 5 seconds
    while 1:
      res = mt.recv(5);
      if res is None :
        print '...retrying to receive after 5 seconds ...'
        # Maybe try to do some other work
        continue
      else :
        # "Do something" with the result
        if res=="EOF" : return
        print res


# Try to open up the MidasTalker, in case not immediately available 
def openup (mt) :
  # Open the file, with a retry in case it's not up right away
  while 1:
    try :
      mt.open()
      print 'Opened connection to host:',mt.host,' port:',mt.port
      break
    except error, val :
      print "Problem:", val
      print "...couldn't open right away, backing off 5 seconds"
      time.sleep(5)

    
if __name__ == '__main__' :

  # Parse arguments
  try :
    opts,args=getopt.getopt(sys.argv[1:],[],["ser=","sock=","arrdisp="])
    if len(args)!=3 : raise error
  except :
    print "Usage: permutation_client.py [--ser=0|1|2|-2|5] [--sock=1|2] [--arrdisp=0|1|2|4] host port array_of_data_to_permute"
    print " Example usage:"
    print '   permutation_client.py --ser=2 localhost 8501 "[1,2,3]"'
    sys.exit(1)

  host = args[0]
  port = string.atoi(args[1])
  data = eval(args[2])
  serialization = SERIALIZE_P0  # Not fast, but backwards compatible
  socket_duplex   = DUAL_SOCKET   # backwards compat, but use single if able
  array_disposition = ARRAYDISPOSITION_AS_LIST   # backwards compat
  
  for opt, val in opts :
    if   opt=="--ser" :     serialization     =  int(val) 
    elif opt=="--sock":
      options = { '1':0, '2':1, '777':777 }
      socket_duplex = options[val]
    elif opt=="--arrdisp" : array_disposition =  int(val)
    else : assert False, "unhandled option"

  # Create, but don't actually connect
  mt = MidasTalker(host, port, serialization, socket_duplex, array_disposition)  

  openup(mt)
  try :
    sendloop(mt, data)
  except error, val :
    print 'Problem:',val
    print 'Server appears to have gone away?  Quitting'


