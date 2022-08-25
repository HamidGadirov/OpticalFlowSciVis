
from midaslistener import *
import sys

BYTE_LEN = 8192   # THESE NEED TO MATCH!!

if len(sys.argv) != 3 :
    print "Usage: host_we_are listening_to port_listening"
    print " ... these should match the UDPClientTable of the M2k server"
    sys.exit(1)
    
host = sys.argv[1]        
port = int(sys.argv[2])
print "host=", host
print "port=", port


ml = MidasListener(host, port, BYTE_LEN, SERIALIZE_P0)
ml.open()

while 1 :
    v = ml.recv(5.0)  # wait 5 seconds
    if v==None :
        print " .. haven't seen anything for 5 seconds"
        continue
    print v



