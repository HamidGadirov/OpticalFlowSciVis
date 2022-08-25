import copy

def CreateBigTable (first=200000, second=100000) :
    t = { }
    a = [ None, 1.0, 'hello', {}, {'a':1}, {'a':1, 'b':2}, [], [1], [1,2], [1,2,3]]
    
    #for ii in xrange(0, 200000) :
    for ii in xrange(0, first) :
        if ii%2==0 :
            t["b"+str(ii)] = copy.deepcopy(a[ii%len(a)])
        else :
            t["a"+str(ii)] = copy.deepcopy(a[ii%len(a)])
    t["nested"] = copy.deepcopy(t)
    #for ii in xrange(0, 100000) :
    for ii in xrange(0, second) :
        t["AAA"+str(ii)] =  "The quick brown fox jumped over the lazy dogs quite a few times on Sunday "+str(ii);

    return t


import sys
first = 200; second = 100
if len(sys.argv)==2 :
    if sys.argv[1]=="small" :
        first = 200; second=100
    elif sys.argv[1]=="medium" :
        first = 2000; second=1000
    elif sys.argv[1]=="large" :
        first = 20000; second=10000
    elif sys.argv[1]=="huge" :
        first = 200000; second=100000

import time

start = time.time()
d = CreateBigTable(first, second)
end = time.time()
print 'Time to create big table', end-start

startx = time.time()
dd = copy.deepcopy(d)
endx = time.time()
print 'Time to deepcopy big table', endx-startx
del dd




import pyobjconvert
pyobjconvert.ConvertToVal([1,2,3], 0)
start2 = time.time()
pyobjconvert.ConvertToVal(d)
end2 = time.time()
print '...time to convert PyObject to Val ...', end2-start2


start3 = time.time()
pyobjconvert.deepcopy(d)
end3 = time.time()
print '...time to convert PyObject to Val and back...', end3-start3


import xmldumper
start0 = time.time()
xmldumper.WriteToXMLFile(d, "/tmp/pythondump.xml", "top");
end0 = time.time()
print 'Time for Python XMLDumper to dump an XML file', end0-start0



start1 = time.time()
pyobjconvert.CWriteToXMLFile(d, "/tmp/cdump.xml", "top")
end1 = time.time()
print 'Time for C XMLDumper to dump an XML file:', end1-start1


print '-----------'

import xmlloader
starts = time.time()
thing1 = xmlloader.ReadFromXMLFile("/tmp/pythondump.xml");
ends = time.time()
print 'Time for Python Loader to load an XML file', ends-starts
# print thing
del thing1

startc = time.time()
thing2 = pyobjconvert.CReadFromXMLFile("/tmp/pythondump.xml");
endc = time.time()
print 'Time for C Ext Loader to load an XML file', endc-startc

