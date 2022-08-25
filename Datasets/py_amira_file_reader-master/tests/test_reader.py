from __future__ import print_function
import py_amira_file_reader.read_amira as read_amira
import os

def get_data_path(fname):
    tests_path = os.path.split( __file__ )[0]
    data_path = os.path.join( tests_path, 'data', fname )
    return data_path

def show_result(data):
    result = data['data']
    print('==='*20, 'RESULT')
    import pprint
    for row in result:
        if 'data' not in row:
            pprint.pprint( row )
        else:
            print('<NOT SHOWING DATA>')
        print()

def test_ascii_surf():
    fname = 'tetrahedron.surf'
    data_path = get_data_path(fname)
    data = read_amira.read_amira( data_path )
    if read_amira.is_debug():
        show_result(data)
    assert data['data'][0]['Parameters']['Materials']['Exterior']['id'] == 1
    assert data['data'][1]['Vertices'].shape == (4,3)

def test_am():
    fnames = ['LHMask.am',
             ]
    for fname in fnames:
        yield check_am, fname

def check_am(fname):
    data_path = get_data_path(fname)
    data = read_amira.read_amira( data_path )
    size = None
    for row in data['data']:
        if 'define' in row:
            size = row['define']['Lattice']
        if 'data' in row:
            arr = row['data']
            if type(size)==int:
                assert len(arr)==size
            else:
                assert arr.shape == tuple(size)
    assert size is not None

def test_parse_ascii_mesh():
    fname = 'hybrid-testgrid-2d.am'
    data_path = get_data_path(fname)
    data = read_amira.read_amira( data_path )
    print('---------------- ascii mesh data -----')
    import pprint
    pprint.pprint(data)
    print('---------------- ascii mesh data done -----')
