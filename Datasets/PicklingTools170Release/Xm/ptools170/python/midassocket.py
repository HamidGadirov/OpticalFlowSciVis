""" The internals of the MidasServer and MidasTalker share a lot of
    code (especially to do primitive reads and primitive writes to
    the socket).  This module encapsulates all that code."""

import socket
from socket import error
import pickle
import cPickle
import struct
import sys
import select
import time
import errno


try :
    from pyocser import ocdumps
    from pyocser import ocloads
except :
    ocdumps = None
    ocloads = None
    
# For select compatibility with Jython
if "Java" in sys.version :
    from select import cpython_compatible_select as system_select
else :
    from select import select as system_select
    
# Different types of Python serialization: Notice that this is
# reasonably backwards compatible with previous releases, and 0 and 1
# still correspond to "Pickling Protocol 0" and "No serialization".
# Now, the int 2 becomes "Pickling Protocol 2".
SERIALIZE_SEND_STRINGS_AS_IS_WITHOUT_SERIALIZATION = 1# No serialization at all
SERIALIZE_PYTHON_PICKLING_PROTOCOL_0 = 0  # Older, slower, more compatible
SERIALIZE_PYTHON_PICKLING_PROTOCOL_2 = 2  # Newer, faster serialization
SERIALIZE_PYTHON_PICKLING_PROTOCOL_2_AS_PYTHON_2_2_WOULD = -2

# Aliases
SERIALIZE_NONE  = SERIALIZE_SEND_STRINGS_AS_IS_WITHOUT_SERIALIZATION
SERIALIZE_P0    = SERIALIZE_PYTHON_PICKLING_PROTOCOL_0
SERIALIZE_P2    = SERIALIZE_PYTHON_PICKLING_PROTOCOL_2
SERIALIZE_OC    = 5 # OpenContainers: there is now a C Extension module

# Older versions of Python 2.2.x specificially don't "quite" work with
# serialization protocol 2: they do certain things wrong.  Before we
# send messages to servers and clients, we need to tell them we are
# using an older Python that does things slightly differently.
SERIALIZE_P2_OLD = SERIALIZE_PYTHON_PICKLING_PROTOCOL_2_AS_PYTHON_2_2_WOULD 

from arraydisposition import *

# Do you use two sockets or 1 socket for full duplex communication?
# Some very old versions (VMWARE) only supported single duplex sockets,
# and so full duplex sockets had to be emulated with 2 sockets.
SINGLE_SOCKET = 0
DUAL_SOCKET   = 1
NORMAL_SOCKET = 777

class MidasSocket_ :
    """ A base class with some primitive I/O facilities for MidasServer
    and MidasTalker """

    once = 0
    def __init__ (self, serialization=SERIALIZE_P0,
                  array_disposition=ARRAYDISPOSITION_AS_LIST) :
        """ Create MidasSocket_ that either Python Pickles or sends strings
        as is.  This also chooses whether or not to use Numeric/numpy or plain
        Python Arrays in serialization. For backwards compat, AS_LIST is 1
        (being true), AS_NUMERIC is 0 (being false) and the new
        AS_PYTHON_ARRAY is 2, AS_NUMPY_ARRAY is 3.

	Note that the Python version of the Midastalker and MidasServer
	CAN NOT currently support ADAPTIVE serialization.  This is
	because inherently, the version of Python only supports
	a few protocols and options natively.  Thus, the talker
	or server is constrained by the serializations available.
	In the future, we will port the M2k serialization and be 
	more careful with array options, but for now the MidasTalker
	and MidasServer (at least in Python) CAN NOT be adative.
	[Although, most Pythons support both 0 and 2 and the protocol
	is adaptive enough to support both of these]"""

        # For backwards compatibility: a true is 1, a false is 0: All
        # enums have been mapped to stay backward compatible, but may need
        # to be mapped from bool to int
        #    1 or True -> SERIALIZE_NONE, 0 or False -> SERIALIZE_P0
        serialization     = int(serialization)
        #   1 or True -> AS_LIST, 0 or False ->AS_NUMERIC
        array_disposition = int(array_disposition)

        
        if array_disposition==2 and self.once==0 :
            c = MidasSocket_()
            c.once = 1
            print "As of PicklingTools 1.2.0, the array disposition" 
            print " AS_PYTHON_ARRAY is deprecated.  You will see this warning"
	    print " only once...." 
            print """
  The whole purpose of adding the ArrayDisposition AS_PYTHON_ARRAY
  was because, in Python 2.6, it was a binary dump: dumping arrays
  of POD (Plain Old Data, like real_4, int_8, complex_16) was blindingly
  fast (as it was basically a memcpy): This was to help Python users who
  did not necessarily have the Numeric/numpy module installed. As of Python 2.7,
  however, the Pickling of Arrays has changed: it turns each element into a
  Python number and INDIVIDUALLY pickles each element(much like the AS_LIST
  option).  The new Pickleloader DOES DETECT AND WORK with both 2.6
  and 2.7 pickle streams, but we currently only dump as 2.6: this change
  in Python 2.7 (and also Python 3.x) defeats the whole purpose of
  supporting array .. we wanted a binary protocol for dumping large amounts
  of binary data!  As of this release, we deprecate the AS_PYTHON_ARRAY
  serialization, but will keep it around for a number of releases.  
        """
         
        self.serialization = serialization
        
        # Find out what this version of Python CAN support
        cap = self.capabilities_()
                
        header = [ 'P', 'Y', '0', '0' ]
        # Array Disposition
        if array_disposition==ARRAYDISPOSITION_AS_PYTHON_ARRAY :
            if not cap["supports_array"] :
                raise error("This version of Python doesn't support array serialization very well (i.e., it's broken in 2.3.4, but appears to work in some versions of Python 2.4, 2.5 and above).  Try setting ArrayDisposition to Numeric (ARRAYDISPOSITION_AS_NUMERIC) or plain Python Lists (ARRAYDISPOSITION_AS_LIST) for your Arrays.")
            else :
                header[2] = 'A'
        elif array_disposition==ARRAYDISPOSITION_AS_NUMERIC : 
            if not cap["supports_numeric"] :
                raise error("This version of Python does not seem to have the Numeric module built-in.  Try building it in, or set the ArrayDisposition to either ARRAYDISPOSITION_AS_PYTHON_ARRAY for Python Arrays (which is as efficient as Numeric arrays) or ARRAYDISPOSITION_AS_LIST for plain Python Lists (less efficient).")
            else :
                header[2] = 'N'
        elif array_disposition==ARRAYDISPOSITION_AS_LIST :
            header[2] = '0'
	elif array_disposition==ARRAYDISPOSITION_AS_NUMPY :
	    if not cap["supports_numpy"] :
	    	raise error("This python does not seem to have the numpy module available.  Install it either from an RPM or yourself.  In the meantime, try ARRAYDISPOSITION_AS_LIST or something else")
	    else :
	    	header[2] = 'U'
        else :
            raise error("Unknown Array Disposition:"+str(array_disposition))

        # Pickling Issues only for protocol 2
        if serialization == SERIALIZE_P2_OLD :
            if cap["supports_new_p2"] :
                raise error("You have requested trying to use Python Pickling Protocol 2 as Python version 2.2.x would:  that won't work.  Your version of Python only pickles the new (correct) way; explicitly set serialization to SERIALIZE_P2 or SERIALIZE_P0")
            else :
                header[3] = '-'  # Forcing servers to recognize backwards
                
        elif serialization == SERIALIZE_P2 :
            if not cap["supports_new_p2"] :
                raise error("You have requested trying to use Python Pickling Protocol 2 as Python version 2.3.x and above would:  that won't work.  Your version of Python only pickles the old way;  explcitly set to AS_PYTHON_2_2")
            else :
                header[3] = '2'
        elif serialization==SERIALIZE_P0 :
            header[3] = '0'
        elif serialization==SERIALIZE_NONE:
            header = []
        elif serialization==SERIALIZE_OC:
            if not cap["supports_oc"] :
                raise error('You are trying to use SERIALIZE_OC, but cannot seem to find it.  You have to make sure the module is built and that the pyocsermodule.so is on your PYTHONPATH.  Try (a) build the module ("cd TOP/PythonCExt; python setup.py") and (b) setenv PYTHONPATH "${PYTHONPATH}:TOP/PythonCExt/build/lib-??????? and (c) try again!')
            header = ['O','C','0','0']
        else :
            raise error("Unknown serialization type arg:"+str(serialization))
        
        # Enough info to finally generate the header
        str_header = ""
        for x in header :
            str_header += x
        self.header = str_header

        # Force shutdown
        self.forceShutdownOnClose = True

    def closing_ (self, fd) :
        """When we want to close a socket, some platforms don't seem
        to respond well to 'just' a close: it seems like a 
        shutdown is the only portable way to make sure the socket dies.
        Some people may spawn processes and prefer to use close only,
        so we allow 'plain' close is people specify it, otherwise
        we force a shutdown.  The default is TRUE."""
        if self.forceShutdownOnClose :
            try :
                fd.shutdown(socket.SHUT_RDWR)
            except :  # Overly exception-happy: may have already closed already
                pass
        fd.close()
            
    

    def packageData_(self, raw_data) :
        """Helper function to pack data (Pickled, raw, etc.).
        and return the unpacked data"""
        if self.serialization == SERIALIZE_NONE :
            data = str(raw_data)
        elif self.serialization == SERIALIZE_OC :
            data = ocdumps(raw_data)
        else :
            data = cPickle.dumps(raw_data, abs(self.serialization))
        return data
            
                
    def unpackageData_(self, packaged_data):
        """Helper function to unpackage the data for the different
        types of serialization (Pickling, raw data, etc.). The
        packaging policy is dictated by the constructor."""
        if self.serialization == SERIALIZE_NONE :
            retval = packaged_data
        elif self.serialization == SERIALIZE_OC :
            retval = ocloads(packaged_data)
        else :
            try :  
                retval = cPickle.loads(packaged_data)
            except ValueError :
                retval = pickle.loads(packaged_data)

        return retval


    def recvBlocking_ (self, fd):
        """Blocking call that returns the message from the socket."""
        bytes = self.readExact_(fd, 4) # Read 4 bytes of message length
        if (bytes=='') :
            return
        (unpack_bytes,) = struct.unpack("!I", bytes)
        # Special ESC code tells us we have more than int_u4 of data
        if unpack_bytes == 0xFFFFFFFF :
            bytes = self.readExact_(fd, 8)
            (unpack_bytes,) = struct.unpack("!Q", bytes)
            
        # Do we have a header?  If we do, do we count it?            
        hdr = self.readExact_(fd, 4) # Is there a header?    
        if (hdr[:2]=='PY' and hdr[2] in 'AN0U' and hdr[3] in '-02') :
            # legal 
            read_bytes = unpack_bytes
            already_read = ""
            received_serialization = SERIALIZE_P2
            if hdr[3]!='2' : received_serialization = SERIALIZE_P0
            
        elif (hdr=="M2BD") :
            # Can't handle these: We will simply ignore
            # The "default" M2k serialization is M2k Binary so it is
            # possible to get an unauthorized
            raise Exception("""
Can't handle Midas 2k binary serialization from Python: Reconnect and seed
the conversation (be the first to send a message) so you can set the
serialization to Python 0 or 2""")
        elif (hdr[:2]=="OC") :
            # OpenContainers, can now handle
            # legal 
            read_bytes = unpack_bytes
            already_read = ""
            received_serialization = SERIALIZE_OC
        else :
            # Assume it's just some raw data
            read_bytes = unpack_bytes - 4
            already_read = hdr
            received_serialization = SERIALIZE_NONE

        data = already_read+self.readExact_(fd, read_bytes) #get message
        return self.unpackageData_(data)


    def readExact_(self, fd, bytes):
        """Read exactly the given number of bytes from the socket
        (blocking)."""
        # Sometimes reads get broken up... it's very rare (or
        # impossible?)  to get an empty read on a blocking call.  If
        # we get too many of them, then it probably means the socket
        # went away.
        retries = 1000
        bytes_to_read = bytes
        final_buff    = ''
        MAX_BYTES = 1073741824
        while bytes_to_read :
            if bytes_to_read > MAX_BYTES :
                byte_chunk = MAX_BYTES
            else :
                byte_chunk = bytes_to_read
            #print 'trying to read', byte_chunk, 'bytes to read=', bytes_to_read
            t = fd.recv(byte_chunk) # socket's recv, not ours!
            if t == "": retries-=1
            if retries<0 :
                # Since this is a blocking read, if there's no data,
                # we should just block.  If we return nothing, that seems
                # to indicate that the server went away
                raise error, "Reading socket seems to have gone away"
            bytes_to_read -= len(t)
            final_buff    += t
        return final_buff

    def sendBlocking_(self, fd, val) :
        """Blocking call to send the val over the socket"""
        data = self.packageData_(val)

        # Normally 4 byte header, but if really large
        # we write special ESC 0XFFFFFFFF then the 8 byte length
        if len(data) >= 0xFFFFFFFF :
            bytes_to_send = struct.pack("!I", 0xFFFFFFFF)
            self.writeExact_(fd, bytes_to_send)
            bytes_to_send = struct.pack("!Q", len(data))
            self.writeExact_(fd, bytes_to_send)
        else :
            bytes_to_send = struct.pack("!I", len(data))
            self.writeExact_(fd, bytes_to_send)
            
        if self.serialization != SERIALIZE_NONE : 
            self.writeExact_(fd, self.header)
        self.writeExact_(fd, data)

    def writeExact_(self, fd, buff):
        """Write exactly what's in the buff to the socket (blocking)."""
        #res = fd.sendall(buff)
        retries = 1000
        bytes_to_write = len(buff)
        where = 0
        MAX_BYTES = 1073741824
        while bytes_to_write :
            if bytes_to_write > MAX_BYTES :
                byte_chunk = MAX_BYTES
            else :
                byte_chunk = bytes_to_write
            # print 'trying to write', byte_chunk
            partial = buff[where:where+byte_chunk]
            bytes_sent = fd.send(partial) # socket's send, not ours!
            if bytes_sent==0: retries-=1
            if retries<0 :
                # Since this is a blocking read, if there's no data,
                # we should just block.  If we return nothing, that seems
                # to indicate that the server went away
                raise error, "Writing socket seems to have gone away"
            bytes_to_write -= bytes_sent
            where += bytes_sent
        pass  
        # print 'Done with sendall', len(buff)


    def capabilities_(self) :
        """Figure out the capabilities for serialization of this
        particular version of Python and generate a table describing those
        capabilities."""

        cap = { "supports_numeric":0, "supports_array":0, "supports_new_p2": 1,
                "supports_numpy":0, "supports_oc":0 }

        # OCserialization?
        cap["supports_oc"] = (ocdumps != None)
            
        # See if we can support Numeric
        try :
            import Numeric   # we hopefully prefer numpy
        except ImportError :
            pass
        else :
            cap["supports_numeric"] = 1
        
        # See if we can support arrays by trying to pickle
        # and see what we get.  So far, Python 2.2 and 2.3
        # don't work with this, but 2.5 does
        res = ""
        try :
            import array
            import cPickle
            a = array.array('i', [1])
            res = cPickle.dumps(a, 2)
        except :
            pass  # Any exception pretty much negates this  
        if res=='\x80\x02carray\narray\nq\x01U\x01iU\x04\x01\x00\x00\x00\x86Rq\x02.' or res=='\x80\x02carray\narray\nq\x01U\x01i]q\x02K\x01a\x86Rq\x03.':
            cap["supports_array"] = 1

        # Python 2.2.x is quirky and doesn't support Protocol 2
        # very well: it serializes LONGs differently, Numeric
        # differently and doesn't have the P2_PREAMBLE
        res = ""
        try :
            import cPickle
            res = cPickle.dumps(111111111111L, 2)
        except :
            pass
        if res=="L111111111111L\n." :
            cap["supports_new_p2"] = 0

        # Numpy?
	try :
	    import numpy
	    numpy_support = 1
	except :
	    numpy_support = 0
	cap["supports_numpy"] = numpy_support

        return cap  # Here are the capabilties of this install

    def dataReady_ (self, read_fd, timeout=0) :
        """ Helper function: This function returns True immediately if
        the given file descriptor (for a socket) is available to read
        without blocking.  If the socket is NOT available, then the
        socket is watched for timeout seconds: if something becomes
        available on the socket in that time, this returns True,
        otherwise the entire timeout passes and this returns False.
        Note that this method allows a user to 'poll' a read socket
        without having to do an actual read.  A timeout of None is
        forever, a timeout of 0 is no wait.  """
        if read_fd < 0 : raise socket.error("dataReady_ called with closed fd")
        (reads,o1,o2) = self.mySelect_([read_fd], [], [], timeout)
        return len(reads)!=0
        

    def mySelect_ (self, reads, writes, excepts, timeout) :
        """
        Helper function:  Do a select, but watch out for the EINTR,
        which is a perfectly valid return code (signifying an
        interrupted signal call, but not necessarily an error in
        the select itself).  When we see an EINTR, simply ignore
        it and continue back into the select
        """
        lhs = ([], [], [])
        while 1 :
            try :
                last_time = time.time()
                lhs = system_select(reads, writes, excepts, timeout)
                break
            except select.error, v :
                if v[0]!= errno.EINTR :
                    raise
                if timeout is not None :
                    current_time = time.time()
                    # print "BEFORE:", timeout
                    timeout -= (current_time - last_time)
                    # print "AFTER:", timeout
                    if timeout <= 0 :
                        break
                else :
                    break
            #except :
            #    import sys
            #    print 'uncaught!', sys.exc_type, sys.exc_value
            #    print "?????????"

        return lhs






