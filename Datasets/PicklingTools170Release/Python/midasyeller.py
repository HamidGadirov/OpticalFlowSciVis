#!/usr/bin/env python

'''This module exists to provide a simplified interface for implementing
   a UDP Server (MidasYeller) talking to to UDP Clients (MidasListener).
   It should feel sort of like Midastalker and MidasServer interface,
   but of course uses UDP sockets instead, as it may be lossy.'''


import socket
from socket import error
import select
from midassocket import *

class MidasYeller(MidasSocket_) :
    """The MidasYeller class: a python implementation of a UDP
    server."""

    def __init__(self, udp_message_length_upper_byte_limit=1024,
                 serialization = SERIALIZE_P0,
                 array_disposition=ARRAYDISPOSITION_AS_LIST) :
        """Initialize the MidasYeller.  Note that we have to manually
        add (using addListener below) listeners before you send data.
        See the MidasTalker for more dicussion on serialization and
        array_disposition."""

        MidasSocket_.__init__(self, serialization, array_disposition)
        self.host_port_map = { }
        self.upper_byte_limit = udp_message_length_upper_byte_limit
        
    def __del__(self) :
        self.cleanUp()

    def cleanUp(self) :
        """Alias for close"""
        for (hp, fd) in self.host_port_map.iteritems() :
            host, port = hp
            # print hp, fd, self.host_port_map
            if fd :
                fd.close()
        self.host_port_map = { }

    def close(self) :
        """Clean up and destroy all sockets"""
        self.cleanUp()
        
    def addListener(self, host, port) :
        """Add a new client (listener) that should be at the given host/port"""
        host_pair_tuple = (host, port)
        self.removeListener(host_pair_tuple)
        fd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)            
        self.host_port_map[host_pair_tuple] = fd

    def removeListener(self, host_pair_tuple) :
        """Stop yelling at a particular listener"""
        if host_pair_tuple in self.host_port_map :
            fd = self.host_port_map  # Get the socket
            fd.close()
            del self.host_port_map[host_pair_tuple]

    def send(self, data) :
        """Send the data to all listeners."""

        # Iterate through all added Hosts and Ports
        for (hp, fd) in self.host_port_map.iteritems() :
            host, port = hp
            # print hp, fd, self.host_port_map
            if fd :
                # TODO: Make this adaptive? Probably not because
                # we have to be backward compatible with the M2k version
                serialized_data = self.packageData_(data)
                mlen = len(serialized_data)
                if mlen > self.upper_byte_limit :
                    mesg = "Message sending is "+str(mlen)+\
                    " bytes, but your hardcoded limit is "+\
                    str(self.upper_byte_limit)   
                    raise ValueError(mesg)
                fd.sendto(serialized_data, hp)
            else :
                print hp,' not currently opened to send data to'
    
