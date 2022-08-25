
# Simple test to try out WaitForReadyMidasTalker.  You need to run a
# MidasServer (midasserver_ex.py for example) on "soc-dualquad1"
# (machine 1) and "brhrts" (machine 2) for this to work.

import time
from midastalker import *
# Simple example to play with WaitForReadyMidasTalker

host_port_list = [("soc-dualquad1", 8888) ,("brhrts", 8888)]

l = []
for (host, port) in host_port_list :
    m = MidasTalker(host, port)
    m.open()
    l.append(m)
print 'All listeners ...',l
time.sleep(1)  # Give people a chance to wake up and communicate


# Return immediately all ready:  if none ready, go through one by one and wait
ready = WaitForReadyMidasTalker(l)
print 'These listeners have something for me!', ready

print 'Taking data off of first ready', ready[0].recv()
print 'Taking data off of second ready', ready[1].recv()

# 10 second timeout, return immediately as soon as one is ready
ready = WaitForReadyMidasTalker(l, True, 10)
print ready


  

