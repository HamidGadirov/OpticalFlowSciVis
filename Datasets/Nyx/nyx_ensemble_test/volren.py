import numpy as np

import yt
import yt.frontends.boxlib

import sys

import yt.frontends.boxlib.data_structures as yt_ds

from pathlib import Path
HOME=str(Path.home())

# fname_data=f'{HOME}/dev/data/yt/Enzo_64/DD0043/data0043'

#basePath=f'{HOME}/dev/data/new_ensemble/LyA'
basePath='.'

fname_datav=[f'{basePath}/run_orig/plt?????', f'{basePath}/run_min/plt?????']

for member_idx, fname_data in enumerate(fname_datav):
    # ts = AMReXDatasetSeries("plt?????", parallel=True)

    # 
    'yt.frontends.boxlib.data_structures.NyxDataset'
    ts = yt.load(fname_data, hint='AMReX')

    for i,ds in enumerate(ts):
        # When you call the load function, yt tries to do very little -- this is designed to be a fast operation, just setting up some information about the simulation. Now, the first time you access the "index" it will read and load the mesh and then determine where data is placed in the physical domain and on disk. Once it knows that, yt can tell you some statistics about the simulation:
        ds.print_stats()

        # yt can also tell you the fields it found on disk:

        print(ds.field_list)
        # And, all of the fields it thinks it knows how to generate:

        print(ds.derived_field_list)

        #print(ds.min(), ds.max())


        # some possible fields: ('gas', 'velocity_x'), ('gas', 'velocity_y'), ('gas', 'velocity_z') ('gas', 'velocity_magnitude') ("gas", "density")

        sc = yt.create_scene(ds, lens_type="perspective")

        source = sc[0]


        for field in [('gas', 'velocity_magnitude'), ("gas", "density")]:

            source.set_field(field)

            sc.save(f'rendering_{field[1]}_{member_idx:02d}_{i:05d}.png', sigma_clip=6)
