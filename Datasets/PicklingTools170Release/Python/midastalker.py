
''' A MidasTalker is a client which talks to a server using sockets.
This module exists to allow a simplified interface to talk to a Midas
system, whether talking to an M2k OpalPythonDaemon or the the
MidasServer in either C++ or Python.

Most problems when using the MidasTalker are due to socket problems
(wrong host port or name, server unexpectedly goes away).  These can
be caught with try/expect blocks.  See the example below.

from midastalker import *
mt = MidasTalker("hostme", 9711)
try :
  mt.open()
except error, val :
  # Probably server is not running on the host, port specified
  print val

try :
  mt.recv(10.0)  # 10 second timeout ...
except error, val :
  # Probably socket unexpectedly went away, even though we were connected
  print val

  
'''

from midassocket import *

class MidasTalker(MidasSocket_) :
  
  """ The MidasTalker class: a python client for talking to a
  MidasServer or an M2k OpalPythonDaemon.

  Host is the name of the machine running the server that we wish to
  connect to, port is the port number on that machine
  
  Serialization should be one of:
    0: SERIALIZE_P0 (use Python Pickling Protocol0: slow but compatible)
    1: SERIALIZE_NONE (send data in strings as is),
    2: SERIALIZE_P2 (use Python Pickling Protocol2: fast but less compat)
   -2: SERIALIZE_P2_OLD  (for old version only, see below for info)
    5: SERIALIZE_OC      (NEW! Uses Python C Extension module)
  We default to SERIALIZE_P0 for backwards compatibility, but strongly
  urge users to use SERIALIZE_P2 for the speed.  Note that we don't
  current support SERIALIZE_M2K or SERIALIZE_OC from Python.
  If we need very large arrays (over2-4G), you have to use SERIALIZE_OC

  dual_socket refers to how you communicate with the server: using one
  bi-directional socket or two single directional sockets.
    0: SINGLE_SOCKET
    1: DUAL_SOCKET
  777: NORMAL_SOCKET
  The dual socket mode is a hack around an old TCP/IP stack problem
  that has been irrelevant for years, but both X-Midas and M2k have to
  support it.  We recommend using single_socket mode when you can, but
  default to dual socket mode because that's what X-Midas and Midas 2k
  default to. Note: A server either supports single or dual socket
  mode: NOT BOTH.

  NORMAL_SOCKET can be used to talk to normal sockets that don't have
  the ridiculous DUAL_SOCKET or SINGLE_SOCKET protocol.

  array_disposition describes how you expect to send/recv binary
  arrays: as Python arrays (import array) or Numeric arrays (import
  Numeric).  By default, the MidasTalker sends the data as lists, but
  this isn't efficient as you lose the binary/contiguous nature.  This
  option lets the constructor check to see if your Python
  implementation can support your choice.  Choices are:
    0: ARRAYDISPOSITION_AS_NUMERIC      (DEPRECATED: only use if using XMPY < 4.0.0)
    1: ARRAYDISPOSITION_AS_LIST         (default)
    2: ARRAYDISPOSITION_AS_PYTHON_ARRAY (recommended Python 2.5 & up)
    3: ARRAYDISPOSITION_AS_NUMERIC_WRAPPER (NOT CURRENTLY SUPPORTED)
    4: ARRAYDISPOSITION_AS_NUMPY        (recommended!!)

  When using SERIALIZE_P2_OLD: Version 2.2.x of Python serializes
  slightly differently than later versions, and we support this (to a
  certain extent).  Stay away from this if you possibly can.
  See the MidasSocket_ documentation for more information.
  
  Note that both the Client (MidasTalker and MidasServer) need
  to match on all 3 of these parameters or they WILL NOT TALK!
  """

  
  def __init__(self, host, port, serialization=SERIALIZE_P0, 
               force_dual_socket=DUAL_SOCKET,
               array_disposition=ARRAYDISPOSITION_AS_LIST) :
    """Initialize the MidasTalker with host and port number.  Connection
    occurs when the open() method is called (see below)."""
    MidasSocket_.__init__(self, serialization, array_disposition)
    self.host = host
    self.port = port

    self.dual_socket = force_dual_socket
    self.s = [ None, None ]  # read and write socket descriptors
                             # 0 = read, 1 = write (like stdin, stdout)

  def cleanUp (self) :
    """Clean up and close file descriptors"""
    # It's important that we immediately assign into the
    # s[0] and s[1] so we can track that resource in case some exceptional
    # activity takes place.
    fd0 = self.s[0]
    self.s[0] = None
    if (fd0 != None) : self.closing_(fd0)

    fd1 = self.s[1]
    self.s[1] = None
    if (fd0 != fd1 and fd1 != None) : self.closing_(fd1)

  def __del__ (self) : self.cleanUp()

  def close(self) : self.cleanUp()

  def _connectTimeOut(self, timeout, fd) :
    """Helper function to allow connect to timeout quicker (the
    standard timeout for connect seems to be minutes).  We do this by
    setting the socket descriptor to NON-BLOCKING for the duration of
    the connect call, do a select, then return back to BLOCKING.  The
    current coding makes it so it defers the timeout error to happen
    in openRead instead: this seems more stable (and allows reconnects
    easier)"""
    # Python 2.3.4 and Python 2.5.1 work differently with respect
    # to settimeout and setblocking: to get this to work on both platforms,
    # we set to "non-blocking" with a timeout of 1 second, and then
    # select on the true timeout.  Python 2.2 simply doesn't have this
    # feature so we call connect directly
    if sys.version_info[0]==2 and sys.version_info[1]>2 : # for 2.3 and up
      fd.settimeout(1.0)   # Some non-zero timeout to set non-blocking
      
    fd.connect((self.host, self.port))
    
    if sys.version_info[0]==2 and sys.version_info[1]>2 : # for 2.3 and up
      rs,ws,es = self.mySelect_([fd], [fd], [], timeout)
      if rs!=[] or ws!=[] : # One of the sockets had something
        # Check the error condition to make sure everything okay
        res = fd.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR, 200)
        import struct
        ii = struct.unpack("i", res)[0]
      else :  # Looks like a timeout happened: rare: usually defer to _openRead
        self.cleanUp() # Stop rest of open for proceeding
        raise socket.timeout, "open time out"
      fd.setblocking(1)    # Back to blocking
      
  def _openRead(self, file_desc, timeout, bytes_expected) :
    """Helper function to make sure when we read the Single socket
    of dual socket protocol first few bytes that we allow for timeouts.
    This either just returns and works, or throws an exception
    if the open fails."""
    readables, _, _ = self.mySelect_([file_desc], [], [], timeout)
    if file_desc in readables :
      data = self.readExact_(file_desc, bytes_expected) 
    else :
      # No socket.timeout in Python 2.2
      if sys.version_info[0]==2 and sys.version_info[1] >2 :
        raise socket.timeout, "Timeout in first stages of socket negotiation"
      else :
        raise socket.error,   "Timeout in first stages of socket negotiation"



    return data
    
  def open(self, timeout=None):
    """With the host and port are set, open up the socket and connect.
    During the MidasSocket/talker open negotations, some bytes are
    sent back and forth:  if things aren't there right away, the
    open can raise a socket.error exception to indicate that the
    accept failed.  If the accept succeeds, but the single socket/dual
    socket negotiation fails, this throws a socket.error.  """
    
    self.cleanUp()  # make sure file descriptors reclaimed if open

    self.s[0] = self.s[1] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self._connectTimeOut(timeout, self.s[0])   # blocking timeout
    
    # Single socket or dual-socket?
    if self.dual_socket == NORMAL_SOCKET :
      self.s[1] = self.s[0]
      # Nothing, just don't do crazy midas socket stuff
      return 
      
    elif self.dual_socket == SINGLE_SOCKET : # Single socket, r/w same socket
      m1 = self._openRead(self.s[0], timeout, 16)
      self.clientExpectingSingleSocket_(m1)
    
    elif self.dual_socket == DUAL_SOCKET:  # Must be dual socket mode
      self.s[1] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self._connectTimeOut(timeout, self.s[1])   # blocking timeout
    
      # Distinguish reader and writer
      m1 = self._openRead(self.s[0], timeout, 16)
      self.clientExpectingDualSocket_(m1, "")
      m2 = self._openRead(self.s[1], timeout, 16)
      if m1=='SEND'*4 and m2=='RECV'*4:
        pass # Read and write descriptors in correct place
      elif m1=='RECV'*4 and m2=='SEND'*4:
        # read and write descriptors need to be swapped
        t = self.s[0]
        self.s[0] = self.s[1]
        self.s[1] = t
      else :
        self.clientExpectingDualSocket_(m1, m2)
        
    else :
      raise error, "Unknown socket protocol? Check your dual_socket param"

  def send(self, val, timeout=None):
    """
    Try to send a value over the socket: If the message can't be sent
    immediately, wait for timeout (m.n) seconds for the the socket to
    be available.  If the timeout is None (the default), then this is
    a blocking call.  If the timeout expires before we can send, then
    None is returned from the call and nothing is queued, otherwise 1
    is returned.  
    """
    # A timeout is specified, use select to wait for socket to be available
    _, writeables, _ = self.mySelect_([], [self.s[1]], [], timeout)
    if self.s[1] in writeables :
      self.sendBlocking(val)
      return 1
    else :
      return None

  def sendBlocking(self, val) :
    self.sendBlocking_(self.s[1], val)

  def recv(self, timeout=None) :
    """
    Try to receive the result from the socket: If message is not
    available, wait for timeout (m.n) seconds for something to be
    available.  If no timeout is specified (the default), this is a
    blocking call.  If timeout expires before the data is available,
    then None is returned from the call, otherwise the val is returned
    from the call.  A socket error can be thrown if the socket goes
    away.
    """
    # Timeout: use select to wait until recv call won't block
    readables, _, _ = self.mySelect_([self.s[0]], [], [], timeout)
    if self.s[0] in readables :
      return self.recvBlocking()
    else :
      return None

  def recvBlocking(self) :
    return self.recvBlocking_(self.s[0])

  def dataReady(self, timeout=0) :
    """ This function returns True immediately if the MidasTalker can
    read immediately without waiting.  If the socket is NOT available,
    then the socket is watched for timeout seconds: if something
    becomes available on the socket in that time, this returns True,
    otherwise the entire timeout passes and this returns False.  Note
    that this method allows a user to 'poll' a MidasTalker without
    having to do an actual read.  A timeout of None is forever, a
    timeout of 0 is no wait.

    Slightly unintuitive: If the other end of the socket goes away,
    THIS STILL RETURNS TRUE!  Only when you do the read can you
    detect that the socket has gone away (see UNIX socket FAQs to
    verify this).
    """
    return self.dataReady_(self.s[0], timeout)


  def errout_(self, m) :
    raise error, m
  
  def clientExpectingDualSocket_ (self, m1, m2) :
    """Check and see that we are expecting DUAL_SOCKET correctly:
    throw an exception with a good error message if they don't match,
    otherwise return ok """

    if m1!="SENDSENDSENDSEND" and m1!="RECVRECVRECVRECV" :
      if m1=="SNGLSNGLSNGLSNGL" :
	mesg = """
	  Your client is configured as DUAL_SOCKET, but your server
	  is configured as SINGLE_SOCKET.  You need to set them both to
	  match, then restart both sides (if you change just the client,
	  you may only need to restart the client).
          """
      else :
	mesg = """
	  Something is wrong: the client is set-up as DUAL_SOCKET
	  but we are getting data which suggests your host is set-up as
	  NORMAL_SOCKET, or maybe even a port that doesn't use the
	  Midastalker/Server/OpalPythonDaemon protocols!
	  Recheck your client and server socket type: 
	     (DUAL_SOCKET, SINGLE_SOCKET, NORMAL_SOCKET)
	  and make sure both client and server match (also check port #).
          """
      self.errout_(mesg)
    
    # The preamble is one of the two expected, what if both are same?
    if m1==m2 :
      mesg = """
	Congratulations!  You have found the DUAL_SOCKET race condition!
	It's an innate problem that occurs fairly rarely.  Your sockets
	are messed up and you need to reset both sides.  If this error
	message occurs constantly, you may need to go to SINGLE_SOCKET
	or NORMAL_SOCKET on both sides (if you interface with old systems,
	SINGLE_SOCKET is your only choice)
        """
      self.errout_(mesg)
    
    # If make it here, connection is okay
    pass

    # raise error,"Unknown header messages from sockets:'"+m1+"' & '"+m2+"'"


  def clientExpectingSingleSocket_ (self, m1) :
    """ Check and see that we are expecting SINGLE_SOCKET correctly:
    throw an exception with a good error message if they don't match,
    otherwise return ok"""

    if m1=="SENDSENDSENDSEND" or m1=="RECVRECVRECVRECV" :
      mesg = """
        Your server seems to be in DUAL_SOCKET mode, and your client
	is in SINGLE_SOCKET mode.  You need to reconfigure both client
	and server to match and RESTART both sides (you may need to only
	restart your client side if you set this client to DUAL_SOCKET)
        """
      self.errout_(mesg)
    elif m1!="SNGLSNGLSNGLSNGL" :
      mesg = """
	Something is wrong: the client is set-up as SINGLE_SOCKET
	but we are getting data which suggests your host is set-up as
	NORMAL_SOCKET, or maybe even a port that doesn't use the
	Midastalker/Server/OpalPythonDaemon protocols!
	Recheck your client and server socket type: 
	   (DUAL_SOCKET, SINGLE_SOCKET, NORMAL_SOCKET)
	and make sure both client and server match (also check port #).
        """
      self.errout_(mesg)
    else :
      # Okay
      pass

    # if m1!='SNGL'*4 : raise error, "Expected Single Socket Protocol"



def DistributedWaitForReadyMidasTalker (midastalker_list, immediate_return,
                                        timeout=None) :
  """ Given a list of MidasTalkers, this returns a list of all 'ready'
  MidasTalkers.  Take the given timeout, distribute it evenly over all
  MidasTalkers so that the maximum wait (if all are unavailable) is
  the given timeout.  If none are ready, a full timeout seconds
  elapse, and an empty list is returned.  With an immediate return,
  this function returns as soon as ANY socket is ready, without it,
  each talkers may wait the full distributed quanta of time.  """

  # Choose between distributing the timeout, and having no timeout.
  if timeout==None :
    distributed_timeout = None
  else :
    distributed_timeout = timeout / len(midastalker_list)
    
  # Gather all MidasTalkers that are available in a list
  result = []
  for mt in midastalker_list :
    if mt.dataReady(distributed_timeout) :
      result.append(mt)
      if immediate_return :   # Get out right away if there's one available
        break
      
  return result


def WaitForReadyMidasTalker (midastalker_list, immediate_return=False,
                             timeout=None):
  
  """ Query a list of MidasTalkers, returning a list of MidasTalkers
  that are 'ready' (where 'ready' means the socket can be read from
  immediately).   If no MidasTalkers are immediately ready, then
  the timeout is distributed over all MidasTalkers until at least
  one MidasTalker is ready.  If immediate_return is True, then this
  returns as soon as ANY MidaTalker is ready, otherwise we must
  look at all MidasTalkers.
  """
  
  # First pass, gather up all that are immediately ready, no waiting.
  res = DistributedWaitForReadyMidasTalker(midastalker_list, False, 0)
  if not res : 
    # Second pass, no one was ready.  Distribute the wait (if there is one)
    # and wait for someone to be available.
    res = DistributedWaitForReadyMidasTalker(midastalker_list,
                                             immediate_return,
                                             timeout)
  return res
    
