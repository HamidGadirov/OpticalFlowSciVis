#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import re
import numpy as np
try:
    # Python 3
    from io import BytesIO as StringIO
except ImportError:
    # Python 2
    from StringIO import StringIO
import zlib
import warnings

import collections

TOKEN_NAME = 'name'
TOKEN_NUMBER = 'number'
TOKEN_STRING = 'string'
TOKEN_OP = 'op'
TOKEN_COMMENT = 'comment'
TOKEN_NEWLINE = 'newline'
TOKEN_COMMA = 'comma'
TOKEN_COLON = 'colon'
TOKEN_EQUALS = 'equals'
TOKEN_Vec3Array = 'Vec3Array'
TOKEN_ENDMARKER = 'endmarker'
TOKEN_BYTEDATA_INFO = 'bytedata_info'
TOKEN_BYTEDATA = 'bytedata'

BINARY_DEFAULT = False # If not specified in initial comment, treat file as binary?

dtypes = {'Vertices':np.float32,
          'Triangles':np.int32,
      }
ARRAY_FIELDS = dtypes.keys()

def get_nth_index( buf, seq, n ):
    """find the index of the nth occurance of seq in buf"""
    assert n>=1
    cur_base = 0
    cur_buf = buf
    for i in range(n):
        this_idx = cur_buf.index(seq)
        idx = cur_base + this_idx
        cur_base += this_idx+len(seq)
        cur_buf = cur_buf[this_idx+len(seq):]
    return idx

def test_get_nth_index_simple():
    buf = 'abcbdb'
    assert buf.index( 'b' )==1
    assert get_nth_index( buf, 'b', 1 )==1
    assert get_nth_index( buf, 'b', 2 )==3
    assert get_nth_index( buf, 'b', 3 )==5

def test_get_nth_index_complex():
    buf = 'aa111bb111cc111'
    assert buf.index( '111' )==2
    assert get_nth_index( buf, '111', 1 )==2
    assert get_nth_index( buf, '111', 2 )==7
    assert get_nth_index( buf, '111', 3 )==12

class Matcher:
    def __init__(self,rexp):
        self.rexp = rexp
    def __call__(self, buf ):
        matchobj = self.rexp.match( buf )
        return matchobj is not None

re_string_literal = re.compile(br'^".*"$')
is_string_literal = Matcher(re_string_literal)

re_bytedata_info = re.compile(br'^@(\d+)(\((\w+),(\d+)\))?$')
is_bytedata_info = Matcher(re_bytedata_info)

re_bytedata_key = re.compile(br'^@(\d+)$')
is_bytedata_key = Matcher(re_bytedata_key)

# from http://stackoverflow.com/a/12929311/1633026
re_float = re.compile(br'^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$')
is_number = Matcher(re_float)

re_name = re.compile(br'^[a-zA-Z0-9_]+(\[\d\])?$')
is_name = Matcher(re_name)

re_quoted_whitespace_splitter = re.compile(br'(".*")|[ \t\n]')

def lim_repr(value):
    full = repr(value)
    if len(full) > 100:
        full = full[:97]+'...'
    return full

def rle_decompress(buf):
    result = []
    idx = 0
    buflen = len(buf)
    while idx < buflen:
        control_byte = ord(buf[idx:idx+1])
        idx += 1
        if control_byte==0:
            break
        elif control_byte <= 127:
            repeats = control_byte
            new_byte = buf[idx:idx+1]
            idx += 1
            result.append( new_byte*repeats )
        else:
            num_bytes = control_byte-128
            new_bytes = buf[idx:idx+num_bytes]
            idx += num_bytes
            result.append( new_bytes )
    final_result = b''.join(result)
    return final_result

class Tokenizer:
    def __init__( self, fileobj ):
        self.buf = fileobj.read()
        self.last_tokens = []
        self.file_info = {}
        self._bytedata = {}
        self.defines = {}
    def add_defines(self, define_dict ):
        self.defines.update(define_dict)
    def get_tokens( self ):
        # keep a running accumulation of last 2 tokens
        for token_enum,token in enumerate(self._get_tokens()):
            self.last_tokens.append( token )
            while len(self.last_tokens) > 3:
                self.last_tokens.pop(0)
            if token_enum==0:
                if token[0] == TOKEN_COMMENT and token[1]=='# HyperSurface 0.1 BINARY':
                    self.file_info = {'type':'HyperSurface',
                                      'version':'0.1',
                                      'is_binary':True}
                elif token[0] == TOKEN_COMMENT and token[1]=='# HyperSurface 0.1 ASCII':
                    self.file_info = {'type':'HyperSurface',
                                      'version':'0.1',
                                      'is_binary':False}
                elif token[0] == TOKEN_COMMENT and token[1]=='# AmiraMesh 3D BINARY 2.0':
                    self.file_info = {'type':'AmiraMesh',
                                      'version':'2.0',
                                      'is_binary':True}
                elif token[0] == TOKEN_COMMENT and token[1]=='# AmiraMesh 3D BINARY-LITTLE-ENDIAN 2.0':
                    self.file_info = {'type':'AmiraMesh',
                                      'version':'2.0',
                                      'is_binary':True}
                elif token[0] == TOKEN_COMMENT and token[1]=='# AmiraMesh 3D ASCII 2.0':
                    self.file_info = {'type':'AmiraMesh',
                                      'version':'2.0',
                                      'is_binary':False}
                elif token[0] == TOKEN_COMMENT and token[1]=='# AmiraMesh BINARY-LITTLE-ENDIAN 2.1':
                    self.file_info = {'type':'AmiraMesh',
                                      'version':'2.1',
                                      'is_binary':True}
                else:
                    warnings.warn('Unknown file type. Parsing may fail.')
            yield token
    def _get_tokens( self ):
        newline = b'\n'
        lineno = 0
        while len(self.buf) > 0:

            if (len(self.last_tokens)>=3 and
                self.last_tokens[-3][0]==TOKEN_NAME and
                self.last_tokens[-3][1] in ARRAY_FIELDS and
                self.last_tokens[-2][0]==TOKEN_NUMBER and
                self.last_tokens[-1][0]==TOKEN_NEWLINE):

                n_elements = int(self.last_tokens[-2][1])

                if self.file_info['type']=='HyperSurface':
                    if self.file_info.get('is_binary',BINARY_DEFAULT):
                        sizeof_element = 3*4 # 3x floats, 4 bytes per float
                        n_bytes = n_elements * sizeof_element

                        this_line, self.buf = self.buf[:n_bytes], self.buf[n_bytes:]
                        lineno += 1

                        assert len(this_line)==n_bytes

                        this_data = parse_binary_data(this_line,dtypes[self.last_tokens[-3][1]])
                        yield ( TOKEN_Vec3Array, this_data, (lineno,0), (lineno, n_bytes), this_line )
                    else:
                        idx = get_nth_index( self.buf, newline, n_elements )
                        this_line, self.buf = self.buf[:idx], self.buf[idx:]
                        lineno += n_elements

                        this_data = parse_ascii_data(this_line)
                        yield ( TOKEN_Vec3Array, this_data, (lineno-n_elements,0), (lineno, None), this_line )
                else:
                    raise NotImplementedError
                continue

            # get the next line -------
            idx = self.buf.index(newline)+1
            this_line, self.buf = self.buf[:idx], self.buf[idx:]
            lineno += 1

            # now parse the line into tokens ----
            if this_line.lstrip().startswith(b'#'):
                yield ( TOKEN_COMMENT, this_line[:-1].decode("utf-8"), (lineno,0), (lineno, len(this_line)-1), this_line )
                yield ( TOKEN_NEWLINE, this_line[-1:].decode("utf-8"), (lineno,len(this_line)-1), (lineno, len(this_line)), this_line )
            elif this_line==newline:
                yield ( TOKEN_NEWLINE, this_line.decode("utf-8"), (lineno,0), (lineno, 1), this_line )
            else:
                parts = re_quoted_whitespace_splitter.split(this_line)
                parts.append(newline) # append newline
                parts = [p for p in parts if p is not None and len(p)]

                maybe_comma_part_idx = len(parts)-2 if len(parts) >= 2 else None

                colno = 0
                for part_idx, part in enumerate(parts):
                    startcol = colno
                    endcol = len(part)+startcol
                    colno = endcol + 1

                    if part_idx == maybe_comma_part_idx:
                        if len(part) > 1 and part.endswith(b','):
                            # Remove the comma from further processing
                            part = part[:-1]
                            endcol -= 1
                            # Emit a comma token.
                            yield ( TOKEN_COMMA, part.decode("utf-8"), (lineno,endcol), (lineno, endcol+1), this_line )

                    if part in [b'{',b'}']:
                        yield ( TOKEN_OP, part.decode("utf-8"), (lineno,startcol), (lineno, endcol), this_line )
                    elif part==newline:
                        yield ( TOKEN_NEWLINE, part.decode("utf-8"), (lineno,startcol-1), (lineno, endcol-1), this_line )
                    elif part==b':':
                        yield ( TOKEN_COLON, part.decode("utf-8"), (lineno,startcol-1), (lineno, endcol-1), this_line )
                    elif part==b'=':
                        yield ( TOKEN_EQUALS, part.decode("utf-8"), (lineno,startcol-1), (lineno, endcol-1), this_line )
                    elif part==b',':
                        yield ( TOKEN_COMMA, part.decode("utf-8"), (lineno,startcol-1), (lineno, endcol-1), this_line )
                    elif is_number(part):
                        yield ( TOKEN_NUMBER, part.decode("utf-8"), (lineno,startcol), (lineno, endcol), this_line )
                    elif is_name(part):
                        yield ( TOKEN_NAME, part.decode("utf-8"), (lineno,startcol), (lineno, endcol), this_line )
                    elif is_string_literal(part):
                        yield ( TOKEN_STRING, part.decode("utf-8"), (lineno,startcol), (lineno, endcol), this_line )
                    elif is_bytedata_info(part) and startcol != 0:
                        # bytedata_info will not start at beginning of line
                        matchobj = re_bytedata_info.match( part )
                        bytedata_id, enc_size, encoding, size = matchobj.groups()
                        if enc_size is not None:
                            esdict = {'encoding':encoding.decode("utf-8"),'size':int(size)}
                        else:
                            esdict = None
                        self._bytedata[bytedata_id]=esdict
                        yield (  TOKEN_BYTEDATA_INFO, part.decode("utf-8"),  (lineno,startcol), (lineno, endcol), this_line )
                    elif is_bytedata_key(part):

                        matchobj = re_bytedata_key.match( part )
                        bytedata_id = matchobj.groups()[0]
                        info = self._bytedata[bytedata_id]
                        if info is not None:
                            encoding = info['encoding']
                            size = info['size']
                        else:
                            assert len(self.defines.keys())==1
                            for key in self.defines:
                                dim = self.defines[key]
                                break
                            if self.file_info.get('is_binary',BINARY_DEFAULT):
                                encoding='raw'
                                assert len(dim)==3
                                size=dim[0]*dim[1]*dim[2]
                            else:
                                size = None
                        shape = self.defines.get('Lattice',None)

                        if self.file_info.get('is_binary',BINARY_DEFAULT):
                            binary_buf, self.buf = self.buf[:size], self.buf[size:]

                            if encoding=='raw':
                                raw_buf = binary_buf
                            elif encoding=='HxZip':
                                raw_buf = zlib.decompress(binary_buf)
                            elif encoding=='HxByteRLE':
                                raw_buf = rle_decompress(binary_buf)
                            else:
                                raise ValueError('unknown encoding %r'%encoding)

                            arr = np.fromstring( raw_buf, dtype=np.uint8 )
                            arr.shape = shape[2], shape[1], shape[0]
                            arr = np.swapaxes(arr, 0, 2)
                        else:
                            # ascii encoded file
                            raw_buf = []
                            line_idx = 0
                            while 1:
                                lsize = self.buf.index(b'\n')+1
                                lbuf, self.buf = self.buf[:lsize], self.buf[lsize:]
                                lbuf = lbuf.strip()
                                if lbuf==b'':
                                    # done with this section
                                    break
                                elements = []
                                for el in lbuf.strip().split():
                                    try:
                                        r = int(el)
                                    except ValueError as err:
                                        r = float(el)
                                    elements.append(r)

                                raw_buf.append( elements )
                                line_idx += 1
                            arr = np.array(raw_buf)

                        yield (  TOKEN_BYTEDATA, {'data':arr},  (lineno,startcol), (lineno, endcol), this_line )
                    else:
                        raise NotImplementedError( 'cannot tokenize part %r (line %r)'%(lim_repr(part), lim_repr(this_line)) )
        yield ( TOKEN_ENDMARKER, '', (lineno,0), (lineno, 0), '' )

def parse_ascii_data(buf):
    s = StringIO(buf)
    return np.genfromtxt(s,dtype=None)

def parse_binary_data(buf,dtype):
    n_bytes = len(buf)
    n_elements = n_bytes//4 # 4 bytes per float/int32
    n_vectors = n_elements//3 # 3 elements per vector
    result = np.fromstring(buf, dtype=dtype)
    result.shape = (n_vectors, 3)
    if sys.byteorder=='little':
        result = result.byteswap()
    return result

def is_debug():
    return bool(int(os.environ.get('DEBUG_AMIRA','0')))

def dbgprn( *args, **kwargs ):
    if is_debug():
        print( *args, **kwargs )

def atom( src, token, tokenizer, level=0, block_descent=False ):
    space = '  '*level
    dbgprn('%sATOM LEVEL %d, token[0]=%r'%(space, level, token[0]))
    end_block = None
    if token[0]==TOKEN_NAME:
        name = token[1]
        dbgprn('%sATOM LEVEL %d name: %r'%(space, level, name))

        if block_descent:
            result = name
        else:
            next_token = next(src)

            if next_token[0] == TOKEN_OP and next_token[1]=='{':
                # this name begins a '{' block
                value, ended_with = atom( src, next_token, tokenizer, level=level+1 ) # create {}
                result = {name: value}

            elif name in ARRAY_FIELDS: # if name in ['Vertices', 'Triangles']:
                # this name begins an array
                assert next_token[0]==TOKEN_NUMBER
                n_vectors = int(next_token[1])
                next_token = next(src)
                assert next_token[0]==TOKEN_NEWLINE
                next_token = next(src)
                assert next_token[0]==TOKEN_Vec3Array
                value = next_token[1]
                assert len(value)==n_vectors
                result = {name: value}

            else:
                # continue until newline
                elements = []
                ended_with = None
                force_colon = False
                while not (next_token[0] == TOKEN_NEWLINE):

                    if next_token[0] == TOKEN_COLON:
                        force_colon = True
                        next_token = next(src)

                    value, ended_with = atom( src, next_token, tokenizer, level=level+1, block_descent=force_colon ) # fill element of []
                    elements.append( value )
                    dbgprn('%sATOM LEVEL %d appended element %r'%(space, level, elements[-1]))
                    if ended_with is not None:
                        break
                    next_token = next(src)

                if ended_with is not None:
                    end_block = ended_with
                else:
                    # loop ended because we hit a newline
                    end_block = 'newline'

                dbgprn('%sATOM LEVEL %d elements 1: %r'%(space, level, elements))
                elements = [e for e in elements if e is not None]
                dbgprn('%sATOM LEVEL %d elements 2: %r'%(space, level, elements))
                if len(elements)==0:
                    result = name
                elif len(elements)==1:
                    result = {name: elements[0]}
                else:
                    result = {name: elements}
    elif token[0] in [TOKEN_COMMENT, TOKEN_COMMA]:
        result = None
    elif token[0] == TOKEN_OP and token[1]=='}':
        result = None
        end_block = 'block'
    elif token[0] == TOKEN_NEWLINE:
        result = None
        end_block = 'newline'
    elif token[0] == TOKEN_OP and token[1]=='{':
        if block_descent:
            raise RuntimeError('descent blocked but encountered block')

        elements = []

        # parse to end of block
        next_token = next(src)
        while not (next_token[0] == TOKEN_OP and next_token[1] == '}'):
            value, ended_with = atom( src, next_token, tokenizer, level=level+1 ) # fill element of {}
            elements.append( value )
            if ended_with=='block':
                break
            next_token = next(src)

        elements = [e for e in elements if e is not None]
        dbgprn('%sATOM LEVEL %d: done, elements %r'%(space, level, elements))
        result = collections.OrderedDict()
        for element in elements:
            if isinstance(element,dict):
                for key in element:
                    assert key not in result
                    result[key] = element[key]
            else:
                assert isinstance(element,type(u'unicode string'))
                assert element not in result
                result[element] = None
    elif token[0]==TOKEN_NUMBER:
        try:
            value = int(token[1])
        except ValueError:
            value = float(token[1])
        result = value
    elif token[0]==TOKEN_STRING:
        value = token[1]
        result = value
    elif token[0]==TOKEN_BYTEDATA_INFO:
        result = None
    elif token[0]==TOKEN_BYTEDATA:
        result = token[1]
    elif token[0]==TOKEN_EQUALS:
        result = None
    else:
        raise ValueError('unexpected token type: %r'%(token[0],))

    dbgprn('%sATOM LEVEL %d: done, returning %r (end_block: %s)'%(space, level, lim_repr(result), end_block))

    return result, end_block

def debugger( src ):
    for x in src:
        space = '  '*20
        if x[0]==TOKEN_BYTEDATA:
            print(space,'TOKEN','bytedata: %d'%(len(x[1]['data'])))
        elif x[0]==TOKEN_Vec3Array:
            print(space,'TOKEN','Vec3Array: shape: %s'%(x[1].shape,))
        else:
            print(space,'TOKEN',x)
        yield x

def read_amira( filename ):
    with open(filename,mode='rb') as fileobj:
        result = read_amira_fileobj( fileobj )
    return result

def read_amira_fileobj( fileobj ):
    """load .surf or .am file"""

    tokenizer = Tokenizer( fileobj )
    src = tokenizer.get_tokens()

    if is_debug():
        src = debugger(src)

    token = next(src)

    result = []
    while token[0] != TOKEN_ENDMARKER:
        this_atom, ended_with = atom(src, token, tokenizer) # get top-level atom
        if this_atom is not None:
            #assert isinstance( this_atom, dict )
            if isinstance( this_atom, dict ):
                if 'define' in this_atom:
                    tokenizer.add_defines( this_atom['define'] )
            result.append( this_atom )
        token = next(src)

    return {'info': tokenizer.file_info,
            'data': result,
            }

def read_surf( fileobj ):
    results = read_amira(fileobj)
    assert results['info']['type']=='HyperSurface'
    return results['data']
