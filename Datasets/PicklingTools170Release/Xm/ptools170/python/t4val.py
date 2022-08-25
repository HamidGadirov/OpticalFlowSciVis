

from primitive import m_grabx
from numpy import array
from numpy import int8
import cPickle

# Example what we get back from T4000 pipe when serialized with SERIALIZE_P2
# a = array([-128,    2,  125,   85,    4,   84,   73,   77,   69,   85,   27, 50,   48,   49,   53,   58,   48,   50,   58,   49,   51,   58, 58,   49,   54,   58,   53,   56,   58,   48,   54,   46,   57, 56,   56,   49,   52,   53,  115,   46], dtype=int8)



def recvT4Tab(hin) :
    """ Grab the T4000 data from a pipe (delivered by sendT4Tab from C++),
        and return the dictionary therein """

    # Grab the data
    data = m_grabx(hin, 1, False)
    try:
       a = data[0][0][1]  # The pickled data
    except:
       return None

    # Convert to string so we can unpickle
    s = ""
    for x in a:
       i = chr(x & 0xff)
       s += i
 
    # unpickle!
    try:
        ret = cPickle.loads(s)
    except:
        ret = None
    return ret


