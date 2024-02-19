from yt.mods import *
from yt.data_objects.api import *
from yt.analysis_modules.halo_finding.api import *

def main():
    pf = EnzoStaticOutputInMemory() 
    halos = parallelHF(pf, resize=False, threshold=80.0)
    halos.write_out("halos_%s.out" % pf)


