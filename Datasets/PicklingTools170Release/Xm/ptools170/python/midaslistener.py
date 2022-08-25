#!/usr/bin/env python

"""A client for a MidasYeller."""

import socket
from socket import error
from midassocket import *

class MidasListener(MidasSocket_) :
    """The MidasListener listens to a single MidasYeller.  If we
    are not actively trying to receive data, it may be lost as the
    MidasYeller sends UDP packets.  See the MidasTalker for more
    information on serialization and array_disposition."""

    def __init__(self, host, port, udp_message_length=1024,
                 serialization = SERIALIZE_P0,
                 array_disposition = ARRAYDISPOSITION_AS_LIST) :
        """Initalize the UDP listener with the given host and port.
        Since we are listening to UDP packets, we have an upper
        limit on how big the packet may be.
        """

        MidasSocket_.__init__(self, serialization, array_disposition)
        
        self.udp_message_length = udp_message_length
        self.host = host
        self.port = port
        self.fd   = None

        # When we do a recvfrom, we get back a transitory socket we
        # can use to talk back to the server
        self.transitory_socket = None

    def __del__(self) :
        self.close()

    def cleanUp(self) :
        """Clean up and close file descriptors"""
        fd = self.fd
        self.fd = None
        if fd != None : fd.close()


    def open(self) :
        # Create the socket and bind it to server,port we want to
        # listen to
        self.cleanUp() 
        
        self.fd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.fd.bind((self.host, self.port))


    def close(self) :
        """Close: alias for cleanUp"""
        self.cleanUp()
        

    def recv(self, timeout=None) :
        """Try to receive a UDP packet from the MidasYeller.  If a
        message is not available, wait for a timeout of (m.n) seconds
        for something to become available.  If no timeout is specified
        (the default), this is a blocking call.  If timeout expires
        before the data is available, then None is returned from the
        call, otherwise the val is returned from the call.  A socket
        error can be thrown if the socket goes away."""

        # Timeout: use select to wait until recv call won't block
        readables, _, _ = self.mySelect_([self.fd], [], [], timeout)
        if self.fd in readables :
            packaged_data, ts = self.fd.recvfrom(self.udp_message_length)
            # TODO: Make this adaptive?  probably not because it has to
            # be compatible with M2k
            raw_data = self.unpackageData_(packaged_data)
            self.transitory_socket = ts 
            return raw_data
        else :
            return None

        



    
    
