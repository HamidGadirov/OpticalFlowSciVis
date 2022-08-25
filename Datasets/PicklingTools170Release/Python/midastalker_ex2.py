#!/usr/bin/env python

# A more complicated example showing how to talk to OpalPythonDaemon
# and MidasServers, but still 
# (a) recover from errors
# (b) use timeouts on receieves
# (c) Choosing Python serialization options (Protocol 0 or 2, etc.)

from midastalker import * # Make sure you have this on your PYTHONPATH
                          # so the import picks this up (or copy it to cd)
import sys
import string
import time
import getopt

import numpy 
# Some testing for shoving Numeric shapeless arrays across the pipe
try :
  import Numeric
  a = Numeric.zeros( (), 'F')
  b = Numeric.zeros( (3), 'F')
except :
  a = []
  b = [0.0, 0.0, 0.0]
  
# No timeout exception in Python 2.2
if sys.version_info[0] == 2 and sys.version_info[1] > 2 :
  socket_timeout = socket.timeout
else :
  socket_timeout = IOError

  
# Main loop where you send and receive messages to the MidasTalker (mt)
def sendloop (mt) :

  loop = 0
  while 1 :
    loop += 1
    # if (loop==13) : sys.exit(1)
    
    # Create a message (a table) and send it on over to OpalPythonDaemon
    shared = { 'shared': 1 }
    t = { 'a':1, 'hello':[3.3, 4.4, 5.5], "nest":{'cc':17} ,'a':a, 'b':b}
    #print t['array']
    #print type(t['array'])
    t['proxy1'] = shared
    t['proxy2'] = shared

    if (loop % 10 == 0) :
      # t["VERYLARGE"] = "1"*((2**32)+1)
      #t["numpy"] = numpy.zeros(2**32+1, 'b')
      pass
      
    print 'Sending request:', t     
    mt.send(t)

    # See if we can receive, wait 5 seconds
    while 1:
      res = mt.recv(5) 
      if res is None :
        print '...retrying to receive after 5 seconds ...'
        # Maybe try to do some other work
        continue
      else :
        # "Do something" with the result
        print 'Got Result:', len(res) # ,res
        break



# Try to open up the MidasTalker, in case not immediately available 
def openup (mt) :

  # Open the file, with a retry in case it's not up right away
  while 1:
    try :
      mt.open(5.0)   # Timeout if can't open right away
      print 'Opened connection to host:',mt.host,' port:',mt.port
      break

    except socket.gaierror, val :
      print 'Had trouble with name: did you mistype the hostname?', val
      print "...couldn't open right away, backing off 5 seconds"
      time.sleep(5)
    except socket.herror, val :
      print 'Host error?  Check your DNS?', val
      print "...couldn't open right away, backing off 5 seconds"
      time.sleep(5)
    except socket_timeout, val :
      print 'Open TIMEOUT: is your server slow, suspended or in a weird state?', val
      print "...couldn't open right away, backing off 5 seconds"
      time.sleep(5)
    except error, val :
      print "Problem:", val
      print "...couldn't open right away, backing off 5 seconds"
      time.sleep(5)


# Dumb routine to make sure Signal USR1 is NOT ignored and does something
# even it is just an empty shell.  
def routine (signum, stackframe) : pass

if __name__ == '__main__' :

  # By putting this in, we can send a signal USR1 to the process to
  # call routine1 (instead of terminating). With this, we can force
  # a EINTR on a select system call with a kill -s USR1 process-id
  import signal
  signal.signal(signal.SIGUSR1, routine)
  
  try :
    opts,args=getopt.getopt(sys.argv[1:],[],["ser=","sock=","arrdisp="])
    if len(args)!=2 : raise error
  except :
    print "Usage: midastalker_ex2.py [--ser=0|1|2|-2|5] [--sock=1|2] [--arrdisp=0|1|2|4] hostname portnumber"
    sys.exit(1)

  host = args[0]
  port = string.atoi(args[1])
  serialization = SERIALIZE_P0  # Not fast, but backwards compatible
  dual_socket   = DUAL_SOCKET   # backwards compat, but use single if able
  array_disposition = ARRAYDISPOSITION_AS_LIST   # backwards compat
  
  for opt, val in opts :
    if   opt=="--ser" :     serialization     =  int(val) 
    elif opt=="--sock":
      options = { '1':0, '2':1, '777':777 }
      socket_duplex = options[val]
    elif opt=="--arrdisp" : array_disposition =  int(val)
    else : assert False, "unhandled option"

  # Create, but don't actually connect
  mt = MidasTalker(host,port,serialization,socket_duplex,array_disposition) 

  # Keep trying to reconnect and work
  while 1:
      openup(mt)
      try :
        sendloop(mt)
      except error, val :
        print 'Problem:',val
        print 'Server appears to have gone away?  Attempting to reconnect'



