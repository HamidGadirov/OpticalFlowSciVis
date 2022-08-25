# Simple example showing how to use the MidasServer Python class

from midasserver import *  # Make sure you have this and midassocket on your
                           # PYTHONPATH so the import picks this up

class MyServer(MidasServer) :
    """ MyServer demonstrates how to write your own MidasServer for
    communicating with MidasTalkers"""
    def __init__ (self, have_server_send_mesg_at_connect,
                  host, port, ser, socket_duplex, dis) :
        MidasServer.__init__(self, host, port, ser, socket_duplex, dis)
        self.count = 0
        
        # If the server sends a message at connect time, that
        # establishes the serialization of the session, which you may
        # or may not want.  If you wish the client to establish the
        # serialization of the session, then set this to false.
        # Otherwise, the host always does.
        self.haveServerSendMessageAtConnect = have_server_send_mesg_at_connect
        
    def acceptNewClient_ (self, read_fd, read_addr, write_fd, write_addr):
        print 'MYSERVER:Connection',read_fd, write_fd
        print '         Made from',read_addr,write_addr

        # Send a message right as it connects
        if (self.haveServerSendMessageAtConnect) :
            print "Sending a test message at connect time: This establishes" 
            print " the serialization for the session. If you wish to let the" 
            print " the client set the serialization, don't send this message."
            test = { 'TEST': 'at connect', 'a' : [1,2,3] }
            try :
                self.sendBlocking_(write_fd, test)
            except Exception:
                # Don't want to bring down the server if sendBlocking_ fails
                print "Trouble writing back to client? Probably disconnected:",
                print " ... continuing and keeping server up."
                # ... do cleanup code before thread leaves ...

    def readClientData_ (self, read_fd, write_fd, data) :
        print 'MYSERVER:Client',read_fd,write_fd
        print '         saw some data',data
        # Send the same data back to the client who sent it
        try :
            self.sendBlocking_(write_fd, data)
        except Exception:
            # Don't want to bring down the server if sendBlocking_ fails
            print "Trouble writing back to client? Probably disconnected:",
            print " ... continuing and keeping server up."

        print '... and sent the same data back!'

        # Show how to allow shutdown
        max_count = 100000;
        self.count += 1
        if (self.count>max_count) :
            print '... saw ', max_count,' messages .. .shutting down'
            self.shutdown()
        
    def disconnectClient_ (self, read_fd, write_fd) :
        print 'MYSERVER:Client',read_fd,write_fd,'disconnected'


import sys
import string
import getopt

try :
    opts,args=getopt.getopt(sys.argv[1:],[],["ser=","sock=","arrdisp=", "server_send_message_at_connect="])
    if len(args)!=2 : raise error
except :
    print "Usage: python midasserver_ex.py [--ser=0|1|2|-2|5] [--sock=1|2|777] [--arrdisp=0|1|2|4] [--server_send_message_at_connect=0|1] host port"
    sys.exit(1)

host = args[0]
port = string.atoi(args[1])
serialization = SERIALIZE_P0  # Not fast, but backwards compatible
socket_duplex   = DUAL_SOCKET    
array_disposition = ARRAYDISPOSITION_AS_LIST   # backwards compat
server_send = 1               # By default

for opt, val in opts :
    if   opt=="--ser" :     serialization     =  int(val) 
    elif opt=="--sock":
        options = { '1':0, '2':1, '777':777 }
        socket_duplex = options[val]
    elif opt=="--arrdisp" : array_disposition =  int(val)
    elif opt=="--server_send_message_at_connect" : server_send = int(val)
    else : assert False, "unhandled option"
        
a = MyServer(server_send, host, port, serialization, socket_duplex, array_disposition)
a.open()

# Sit in some loop
import time
while 1 :
    time.sleep(1)


# When determine its time to go away, shutdown and then wait
a.shutdown()
a.waitForMainLoopToFinish()



    
