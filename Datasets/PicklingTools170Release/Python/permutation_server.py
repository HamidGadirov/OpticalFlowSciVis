
from __future__ import generators


# More complex example demonstrating how to use the MidasServer
# class.  This example shows how to start a thread for each request
# and have them active waiting to handle requests.
# 
#  There is a major difference between the C++ and Python version:
# Python doesn't have truly concurrent threads (more like pre-emptive
# single processor threads), but C++ does.  So, if you need true
# concurrency, you should be using the C++ MidasServer.


from midasserver import *  

def permutations (L) :
    
    """ Simple way to compute permutations from net(Connelly Barnes)"""
    if len(L) == 1:
        yield [L[0]]
    elif len(L) >= 2:
        (a, b) = (L[0:1], L[1:])
        for p in permutations(b):
            for i in range(len(p)+1):
                yield p[:i] + a + p[i:]



class PermutationServer(MidasServer) :

    """ To write your own socket socket, you inherit from MidasServer,
    and there is a callback event whenever one of three events
    happen:  (1) New client connects 
             (2) Client has a request/data for the server 
             (3) Client disconnects
    The MidasServer runs in a thread (yes a *thread*) that wakes up and
    animates the callback whenever one of the above events occurs.

    In this example, whenever a new client connects, we create a new
    thread to handle the processing.  The client then sends an Arr
    (such as [1,2,3] ) and the client responds with n!  responses,
    where each of the responses are all the different permutations of
    the table (n is the length of the table, so there are n! different
    permutations).  In the case of the request being [1,2,3], the
    server would respond 6 times with [1,2,3], [2,1,3], [3,2,1],
    [1,3,2], [3,1,2], [2,3,1] then disconnect the client."""


    def PermutationMainLoop (self, data) :
        """This is a simple MainLoop which serves out the permutations of
        its data in order to a client until it finishes.  Each client
        gets its own thread which computes permutations."""
        # Capture all the relevant data in local variables
        (read_fd, write_fd, server, original_data) = data

        try :
            # Let's permute the Array given to us
            p = permutations(original_data);
            for x in p :  # Advance to the next Permutation
                server.sendBlocking_(write_fd, x) # ... send it out
            server.sendBlocking_(write_fd, "EOF")
        except Exception:
            # Don't want to bring down the server if the sendBlocking_ fails
            print "Trouble writing back to client? Probably disconnected:",
            print " ... continuing and keeping server up."
            # ... do cleanup code before thread leaves ...

        
    def __init__ (self, host, port, ser, socket_duplex, dis) :
        """Constructor: Running the PermutationServer on this host, port"""
        MidasServer.__init__(self, host, port, ser, socket_duplex, dis) 
        self.clientList = { }
        
    def acceptNewClient_ (self, read_fd, read_addr, write_fd,write_addr):
        """Create a new thread placeholder to be associated with this
        client.  All we know is that a socket connected, so we can't
        actually start the thread to start processing until we get the
        first request."""
        self.clientList[read_fd] = None    
        print "Permutations:Connection (",read_fd,",",write_fd,")"

    def readClientData_ (self, read_fd, write_fd, data) :
        """ Read client data"""
        # Clients are mapped to their thread by the read_fd        
        print 'Permutations:Client (',read_fd,",",write_fd,")"
        # Make sure legal request
        if type([])!=type(data) :
            print 'Warning!  Can only send Lists of data to the Permutation'
            print " server.  Your request of ",data,"is ignored."
            return
        if read_fd in self.clientList and self.clientList[read_fd]!=None :
            pass # Don't do anything if get another request from same client
        else :
            self.clientList[read_fd] = thread.start_new(PermutationServer.PermutationMainLoop, (self,(read_fd, write_fd, self, data),))


    def disconnectClient_ (self, read_fd, write_fd) :
        del self.clientList[read_fd]
        print 'Permutations:Client (',read_fd,",",write_fd,")"
        print '           disconnected.'



import sys
import string
import getopt

try :
    opts,args=getopt.getopt(sys.argv[1:],[],["ser=","sock=","arrdisp="])
    if len(args)!=2 : raise error
except :
    print "Usage: python permutation_server.py [--ser=0|1|2|-2|5] [--sock=1|2|777] [--arrdisp=0|1|2|4] host port"
    sys.exit(1)

host = args[0]
port = string.atoi(args[1])
serialization = SERIALIZE_P0  # Not fast, but backwards compatible
socket_duplex = DUAL_SOCKET   # backwards compat, but use false (single)if able
array_disposition = ARRAYDISPOSITION_AS_LIST   # backwards compat

for opt, val in opts :
    if   opt=="--ser" :     serialization     =  int(val) 
    elif opt=="--sock":
        options = { '1':0, '2':1, '777':777 }
        socket_duplex = options[val]
    elif opt=="--arrdisp" : array_disposition =  int(val)
    else : assert False, "unhandled option"

# All parsed, create and open
a = PermutationServer(host, port, serialization, socket_duplex, array_disposition)
a.open()


# Sit in some loop
import time
while 1 :
    time.sleep(1)


# When determine its time to go away, shutdown and then wait
a.shutdown()
a.waitForMainLoopToFinish()



    
