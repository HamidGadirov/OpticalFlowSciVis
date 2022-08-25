import os, tempfile, shutil
from test_reader import get_data_path
from py_amira_file_reader.surf_to_obj import surf_to_obj

def test_convert_surf():
    fname = 'tetrahedron.surf'
    data_path = get_data_path(fname)

    outdir = tempfile.mkdtemp()
    try:
        output_filename = os.path.join(outdir,fname)+'.obj'
        surf_to_obj(data_path, output_filename)
        actual = open(output_filename,mode='r').readlines()
    finally:
        shutil.rmtree(outdir)

    expected = ['v -1.0 -1.0 -1.0\n',
                'v 1.0 1.0 -1.0\n',
                'v 1.0 -1.0 1.0\n',
                'v -1.0 1.0 1.0\n',
                'f 1 2 3\n',
                'f 3 2 4\n',
                'f 4 2 1\n',
                'f 1 3 4\n']

    assert len(expected)==len(actual)
    for i in range(len(expected)):
        assert expected[i]==actual[i]
