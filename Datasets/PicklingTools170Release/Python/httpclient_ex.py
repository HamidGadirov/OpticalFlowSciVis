
import httplib
import time

conn = httplib.HTTPConnection("localhost", 8888)
conn.set_debuglevel(1)
while 1 :

  conn.request("GET", "/index.html" )
  r1 = conn.getresponse()
  print r1.isclosed()
  print r1.status, r1.reason
  data1 = r1.read()
  print data1

  data2 = r1.read()
  print data2
  
  conn.close()

  
