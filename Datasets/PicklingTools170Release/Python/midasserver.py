
'''

This module exists to allow running a server which can connect to
multiple clients.  It implements a server, emulating what an
OpalPythonDaemon does (from M2k).  The current incarnation does not
support adaptive serialization, but does allow a choice of Protocols:
Python Pickling Protocol 0,2 or no serialization.  See the MidasTalker
class for more details. (Adaptive serialization is slightly problematic
because it requires stealing C++ code from M2k which means that this
module would be more difficult to install because it would require C++
code and no longer be "only Python".  TODO: We hope to rectify this.)

The MidasServer supports both dual-socket and single socket mode.
Dual-socket mode exists for compatibility with very old sockets
libraries, and is the default mode for the M2k OpalPythonDaemon, thus
it is the default mode for the MidasServer.  If you can, you probably
want to use single socket mode, as frequent connects/disconnects
exposes a race condition in dual-socket mode.

The typical user will inherit from MidasServer, overriding the
methods 

    acceptNewClient_    # Callback for when new client appears 
    readClientData_     # Callback for when data appears for a client
    disconnectClient_   # Callback for when a client disconnects

The callback methods all pass a file descriptor to the user, which can
be used for a blockingSend back to the client.

'''

import select
from midassocket import *
import thread
import time


class MidasServer(MidasSocket_) :
    """ The MidasServer class:  A base class for users to inherit
        from to implement a server.  
    """

    def __init__ (self, host, port, serialization = SERIALIZE_P0,
                  force_dual_socket=DUAL_SOCKET,
                  array_disposition=ARRAYDISPOSITION_AS_LIST):
        """Initialize the MidasServer with host and port number.  This
        doesn't open the server for business yet: see the open() method."""
        MidasSocket_.__init__(self, serialization, array_disposition)
        
        self.host = host
        self.port = port
        self.dual_socket = force_dual_socket
        self.s = None          # read and write socket descriptors
                               # 0 = read, 1 = write (like stdin, stdout)

        self.mainloop_stop   = 0
        self.mainloop_done   = 0
        self.mutex           = thread.allocate_lock()
        self.mainloop_mutex  = thread.allocate_lock()
        self.read_socks      = []
        self.write_socks     = []
        self.client_fd_table = { }

    def __del__ (self) :
        self.shutdown()
        self.waitForMainLoopToFinish()
        # thread.deallocate_lock(self.mutex)
        # thread.deallocate_lock(self.mainloop_mutex)

    def cleanUp (self) :
        """ Clean-up, disconnect everyone, and close file descriptors.
        This can be called over and over to set-up the server."""
        
        # Close all clients
        for client in self.client_fd_table.copy() : 
            self.removeClient(client)
        if self.client_fd_table != { } :
            raise error, "Internal Error: Shouldn't be any clients"

        # Close main door
        if (self.s) : 
            self.closing_(self.s)
            self.s = None
        self.read_socks = []

    def close(self) : self.cleanUp()

    
    def open (self, timeout=1.0) : 
        """ Create a server thread that will start accepting clients.
        Timeout of None will block until some activity is detected on
        the socket.

        Note that we've changed the default on the open: This is how
        long the accept watches the the sockets inside the mainloop
        before it checks for shutdown events: if it's too long, it's
        hard for the server to recognize external events: too short,
        and it polls and wastes CPU.  NONE is almost always the wrong
        value, as it will never wakeup and look for the shutdown
        message: So we make 1.0 be the default: every 1.0 second, the
        accept wakes up, checks shutdown, then goeas back to watching
        the sockets.
        """

        self.cleanUp() # Clean-up file descriptors and reinitialize
        
        # Create the socket that we will be listening on
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)# Allow immediate reopen
        self.s.bind((self.host, self.port))
        self.s.listen(5)


        # Initialize
        self.timeout = timeout

        # A list of client file descriptors: Whenever a client
        # connects or disconnects, it is updated.  Structure:
        # { read_file_desc1 : (address1, write_file_desc1, address_write1),
        #   read_file_desc2 : (address2, write_file_desc2, address_write2),
        #   ... }
        # In single socket mode, since the read_file_desc is the same
        # as the write_file_desc, it is simply a duplicate.
        self.client_fd_table = { }

        # This is a list of currently active read file descriptors.
        # When a new client attaches, it goes to the end of the list.
        # When a client does away, it is deleted.  Note that the
        # "listen" port is always on the read_socks so we can listen
        # for new clients.
        self.read_socks = [self.s]
        
        # Create a thread that watches that socket
        thread.start_new(MidasServer.mainloop, (self,) )

    def now (self) :
        return time.ctime(time.time())


    def socketAccept (self, fd, preamble) :
        """ Helper code for when a socket connects: it connects and sends
        out the appropriate preamble (SNGL, RECV or SEND)*4.  In the case
        of single-socket mode, this is only called once, as the same socket
        is used for read/write.  In dual-socket, it is called twice:
        one accept for what will be the read socket, one accept for
        what will be the write socket."""
        (connection_fd, address) = fd.accept()
        if preamble is not None :
            self.writeExact_(connection_fd, preamble)
        return (connection_fd, address)
        
    def newClientConnect (self, fd) :
        """ A client has appeared and connected: perform maintenance and
        call user hook.  Users override the acceptNewClient_ routine to
        get the hook.
        """
        try :
            # New Client attempting connect
            if self.dual_socket == NORMAL_SOCKET :
                (read_fd, read_addr)   = self.socketAccept(fd, None)
                (write_fd, write_addr) = (read_fd, read_addr)
                
            elif self.dual_socket == DUAL_SOCKET:
                # This is why there is a potential race condition:  We do
                # two accepts right after each other.  This is why we suggest
                # you use single socket mode.
                (read_fd, read_addr)   = self.socketAccept(fd, 'RECV'*4)
                (write_fd, write_addr) = self.socketAccept(fd, 'SEND'*4)
                
            elif self.dual_socket == SINGLE_SOCKET : # Single socket
                (read_fd, read_addr) = self.socketAccept(fd, 'SNGL'*4)
                (write_fd, write_addr) = (read_fd, read_addr)
            else :
                raise error, "Unknown socket protocol?  Needs to be DUAL_SOCKET, SINGLE_SOCKET or NORMAL_SOCKET"
            
            # User hook
            #print 'acceptNewClient_'
            self.acceptNewClient_(read_fd, read_addr, write_fd, write_addr)
        except socket.error :
            print 'Problems with connection??? Aborting connection'
            self.closing_(read_fd)
            self.closing_(write_fd)
            return
        
        self.mutex.acquire() # Lock            
        self.client_fd_table[read_fd] = (read_addr, write_fd, write_addr)
        self.read_socks.append(read_fd) # List of sockets to select
        self.mutex.release() # Unlock


    def newClientData (self, fd) :
        """ A client has gotten data: perform maintenance and call user hook.
        Users override the readClientData_ routine to get the hook.
        """
        data = self.recvBlocking_(fd)
        read_fd = write_fd = fd
        if self.dual_socket :
            self.mutex.acquire() # Lock
            (_, write_fd, _) = self.client_fd_table[read_fd]
            self.mutex.release() # Unlock
        self.readClientData_(read_fd, write_fd, data)

    def removeClient (self, fd) :
        """ A client has gone away.  Remove him from internal lists
            and cleanly close down its connection. """
        
        self.mutex.acquire()  # Lock
        # Remove from list containing active "read" file descriptors
        del self.read_socks[self.read_socks.index(fd)]
        # Remove from list containing list of file descriptors
        entry = self.client_fd_table[fd]  # .. keep copy of for close     
        del self.client_fd_table[fd] 
        self.mutex.release()  # Unlock

        read_fd = fd; write_fd = entry[1]
        if not self.mainloop_done :
            self.disconnectClient_(read_fd, write_fd) # User hook

        # Don't close until all done just in case client needs
        if self.dual_socket :
            self.closing_(read_fd)
            self.closing_(write_fd)
        else :
            self.closing_(read_fd)
            
    def mainloop (ms) :
        """ Mainloop for looping through and watching the socket. """
        ms.mainloop_mutex.acquire() # Keep wait out

        ms.mainloop_done = 0
        while 1:
            
            ms.mutex.acquire() # Lock
            # Do a FULL COPY so we only have to lock this transaction
            rs, ws = ms.read_socks[:], ms.write_socks[:]
            done = ms.mainloop_stop  
            ms.mutex.release() # Unlock

            if done : break
        
            timeout = ms.timeout
            #timeout = 1
            #print 'MAIN LOOP: before select', timeout, rs, ws
            readables,writeables,exceptions=ms.mySelect_(rs, ws, [], timeout)
            #print 'MAIN LOOP: after select', timeout, readables, writeables, exceptions
            
            for r in readables :
            
                is_new_client = (r==ms.s) # Main port
            
                if is_new_client:         # New Client Asking To Connect
                    #print 'new client'
                    ms.newClientConnect(r)
                else :                    # Client with Data to be read
                    try :
                        #print 'new client data'
                        ms.newClientData(r)
                    except error, e :
                        # An error, client went away, take off list
                        #print 'remove client'
                        ms.removeClient(r)

            for w in writeables :
                pass#print "Shouldn't be any writables:",w

            for e in exceptions :
                pass#print "Shouldn't be any exceptions:",e

                
        # Finish main loop
        ms.mainloop_done = 1
        ms.mainloop_mutex.release() # All done with mainloop

    def waitForMainLoopToFinish (self) :
        """ Wait for the mainloop to finish. """
        while not(self.mainloop_done) : 
            self.mainloop_mutex.acquire()
            self.mainloop_mutex.release() # really want condvars here...
        self.cleanUp()

    def shutdown (self) :
        """ Asynchronously tell the server to shutdown.
        Call waitForMainLoopToFinish to make sure it has finished. """
        self.mutex.acquire() # Lock
        self.mainloop_stop = 1
        self.mutex.release() # Unlock
        
    def acceptNewClient_ (self, read_fd, read_addr, write_fd, write_addr) :
        """ User Callback for when a new client is added.  Each client
        can be distiguished by its read_fd (a unique number for each
        client).  The read_addr (and write_addr) gives more
        information about the connection: It's a 2-tuple of address
        and info.  In dual-socket mode, the read_fd and write_fd are
        different numbers, in single-socket mode, they are the
        same."""
        pass

    def readClientData_ (self, read_fd, write_fd, data) :
        """ User Callback for when client data is posted.  The read_fd
        identifies the client that sent the data.  The write_fd is the
        file descriptor to use to send data back to that client.  In
        dual-socket mode, the read_fd and write_fd are different
        numbers, in single-socket mode, they are the same.  Use
        'self.sendBlocking_(write_fd, result)' to send data back to the
        client as a response."""
        pass

    def disconnectClient_ (self, read_fd, write_fd) :
        """ User Callback for when a client disconnects.  The read_fd
        distiguishes which client is disconnecting.  The read_fd and
        write_fd will be closed almost immediately after this method
        returns (the user does not have to close the descriptors)."""
        pass

            
            
