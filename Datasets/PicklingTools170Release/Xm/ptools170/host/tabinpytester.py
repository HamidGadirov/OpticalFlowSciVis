#! /usr/bin/xmpy

""" Simple example showing how a Python X-Midas primitive can read
    T4000 files into dictionaries easily. """

from XMinter import *
from primitive import *
from t4val import recvT4Tab

hin = m_open(m_pick(1), 'r')
m_sync()

while not mcbreak() :
    dictionary = recvT4Tab(hin)
    # DON'T HAVE TO DO Grab!  The recvT4Tab does the grab for you
    #data = m_grabx(hin, 1, False)
    print dictionary

m_close(hin)
