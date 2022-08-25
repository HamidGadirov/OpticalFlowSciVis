# The old version of opalfile is hardcoded for Numeric,
# and the new version is hardcoded for numpy.  This simply
# chooses between the two (i.e., we try to preserve
# some measure of backwards compatibility).
# For a full helpfile, use the particular version:
#  import opalfile_numeric
#  help(opalfile_numpy)

try :
    import numpy
    from opalfile_numpy import *
except :
    try :
        import Numeric
        from opalfile_numeric import *
    except :
        print 'Neither Numeric or numpy is available for opalfile. Abort!'
        throw
