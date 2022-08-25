#!/usr/bin/env python
from __future__ import print_function

import py_amira_file_reader.read_amira as read_amira
import numpy as np
import sys, os

import nrrd # called pynrrd on PyPI

import argparse

def escape(value):
    valstr = str(value)
    if ',' in valstr:
        if '"' in valstr:
            raise NotImplementedError('cannot escape string with internal quotes')
        valstr = '"' + valstr + '"'
    return valstr

def to_csv(csv_data,csv_fname):
    colnames = csv_data.keys()
    with open(csv_fname,mode='w') as fd:
        fd.write( ','.join( map(escape, colnames) ) + '\n' )
        idx = 0
        while True:
            if idx >= len(csv_data[colnames[0]]):
                break
            line_values = []
            for colname in colnames:
                value = csv_data[colname][idx]
                line_values.append( value )
            fd.write( ','.join(map(escape, line_values)) + '\n' )
            idx += 1

def convert_file(fname,csv_fname,nrrd_fname):
    data = read_amira.read_amira( fname )
    dlist = data['data']
    merged = {}
    for row in dlist:
        merged.update(row)
    if 'data' not in merged:
        print('Only binary .am files are supported',file=sys.stderr)
        sys.exit(1)
    arr = merged['data']

    csv_data = {'id':[],
                'name':[],
                }

    # This is true in the beginning as there may be materials (like "Exterior")
    # with no Id.
    ok_to_guess_ids = True

    for name_enum,name in enumerate(merged['Parameters']['Materials'].keys()):
        this_id = None
        expected_id = name_enum+1
        if ok_to_guess_ids:
            this_id = expected_id
        this_dict = merged['Parameters']['Materials'][name]
        if 'Id' in this_dict:
            this_id = this_dict['Id']
            ok_to_guess_ids = False # No longer allow guessing Ids
            assert this_id not in csv_data['id']
        csv_data['id'].append(this_id)
        csv_data['name'].append(name)
    to_csv(csv_data,csv_fname)
    nrrd.write(nrrd_fname, arr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', type=str, help='The file to show')
    args = parser.parse_args()

    fname = args.FILE
    csv_fname = fname+'.csv'
    nrrd_fname = fname+'.nrrd'

    for test_fname in [csv_fname, nrrd_fname]:
        if os.path.exists(test_fname):
            print('ERROR: will not overwrite output file %r'%test_fname, file=sys.stderr)
            sys.exit(1)

    convert_file( fname, csv_fname, nrrd_fname)

if __name__=='__main__':
    main()
