#!/usr/bin/env python
import sys
import py_amira_file_reader.read_amira as read_amira

def str_from_vec(vec):
    return ' '.join([ repr(vi) for vi in vec ])

def write_verts(fd, arr, key):
    for row in arr:
        fd.write( key + ' '+ str_from_vec(row) + '\n' )

def surf_to_obj(input_filename, output_filename):
    results = read_amira.read_amira( input_filename )
    assert results['info']['type']=='HyperSurface'
    with open(output_filename,mode='w') as fd:
        for row in results['data']:
            if 'Vertices' in row:
                write_verts(fd, row['Vertices'],'v')
            if 'Triangles' in row:
                write_verts(fd, row['Triangles'],'f')

if __name__=='__main__':
    input_filename = sys.argv[1]
    obj_filename = input_filename + '.obj'
    surf_to_obj(input_filename, obj_filename)
