"""
Python interface to Midas 2K OPAL file format. Uses the mite or
M2kinter module for read/write paths if available. Supports data in
IEEE big and little endian formats only.

Supports bit data only if t is accessed on byte boundaries and
returns it as a single string.

An Opal file is imported as a dictionary with two keys:
'HEADER' and 'DATA'.

The header is imported as a dictionary.

Continuous data is imported as a list, where each element of the list
could be a time packet (a dictionary), an event data packet (a dictionary),
or continuous data (a list of numpy arrays, or strings if it is bit data).

All multi-element numeric data is returned as a numpy array.
Complex double and float data (CD and CF) are supported directly by
the numpy module; all other complex integer types are expressed by
shaping the returned numpy array. Vectors of time values are
returned as numpy arrays of XMTime objects.

On platforms that don't support 8 byte integers, 8 byte integers are
by default returned as Python long objects. The following options
change that behavior:

LONG_LONG_TO_INT: cast 8 byte integers to ints
LONG_LONG_TO_DOUBLE: cast 8 byte integers to doubles

Using the above options will also modify the header dictionary such
that subsequent writing of the data will write the casted data.

Typical usage:

    import opalfile

    o = opalfile.read('file1')
    # change o['HEADER'] and/or o['DATA']
    opalfile.write('file2', o)
"""

import re
import numpy as N
import struct
import os
import copy
import cStringIO
try :
    from lgm.util.pytime import XMTime
except :
    import XMTime

LONG_LONG_TO_INT = 1 << 0
LONG_LONG_TO_DOUBLE = 1 << 1

_long_long_convert = {
    'LCUX': 'CUL',
    'LUX': 'UL',
    'LX': 'L',
    'DCUX': 'CD',
    'DUX': 'D',
    'DX': 'D',
    }

# per Stephan Terre, default untyped values to float
_untyped_default = float

_indent = ' ' * 4

_struct_endian = {
    'IEEE': '>',
    'EEEI': '<'
    }

if os.sys.byteorder == 'little':
    _native_rep = 'EEEI'
else:
    _native_rep = 'IEEE'

class Link:
    """
    Class for holding link information
    """
    def __init__(self, path):
        self.path = path

class OpalFileException(Exception):
    """
    Class for raising exceptions in this module
    """

class TrackInfo:
    """
    Class for holding track information
    """
    def __init__(self):
        self.elems_per_sample = None
        self.elems_per_block = None
        self.total_elements = None
        self.m2k_format = None
        self.num_format = None
        self.sample_size = None
        self.track_block_size = None
        self.samples_per_block = None
        self.dimensions = None
        self.name = None

class EventTrackInfo:
    """
    Class for holding event track information
    """
    def __init__(self):
        self.fields = []
        self.size = 0
        self.packet_size = 0

class EventFieldInfo:
    """
    Class for holding event field information
    """
    def __init__(self):
        self.name = None
        self.m2k_format = None
        self.elems_per_sample = None
        self.sample_size = None
        self.num_format = None

class DataGrabber:
    """
    Class for grabbing continuous data from the data list (which possibly
    contains multiple different packet types) one block at a time.
    """

    def __init__(self, data, track_info, granularity):
        self.__data = data
        self.__track_info = track_info
        self.__granularity = granularity
        self.__curr_index = 0
        self.__curr_offset = 0

    def grab_block(self):
        ret = []
        for ti in self.__track_info:
            ret.append([])
        remaining_samples = self.__granularity
        for self.__curr_index in xrange(self.__curr_index, len(self.__data)):
            if remaining_samples <= 0:
                break
            item = self.__data[self.__curr_index]
            if isinstance(item, list):
                length = len(item[0])
                if self.__track_info[0].m2k_format == 'BIT':
                    length *= 8
                total_elements = ((length - self.__curr_offset) *
                                  self.__track_info[0].elems_per_sample)
                total_samples = (total_elements /
                                 self.__track_info[0].elems_per_sample)
                assert(total_elements %
                       self.__track_info[0].elems_per_sample == 0)
                start = self.__granularity - remaining_samples
                num_samples = min(total_samples, remaining_samples)
                remaining_samples -= num_samples
                end = self.__granularity - remaining_samples
                for j in xrange(0, len(item)):
                    ti = self.__track_info[j]
                    item_start = self.__curr_offset * ti.elems_per_sample
                    item_end = ((self.__curr_offset + num_samples) *
                                ti.elems_per_sample)
                    start_elem = start * ti.elems_per_sample
                    end_elem = end * ti.elems_per_sample
                    assert(item_end - item_start == end_elem - start_elem)
                    if ti.m2k_format == 'BIT':
                        if end_elem % 8 or start_elem % 8:
                            raise OpalFileException('Cannot split BIT data ' +
                                                    'on a non-byte boundary')
                        end_elem /= 8
                        start_elem /= 8
                        ret[j].append(item[j][item_start:item_end])
                    else:
                        ret[j].append(item[j].flat[item_start:item_end])
                if total_samples <= (remaining_samples + num_samples):
                    self.__curr_offset = 0
                else:
                    self.__curr_offset += num_samples

        for i in xrange(0, len(self.__track_info)):
            ret[i] = N.concatenate(ret[i])

        return ret

class TimeDataGrabber:
    """
    Class for grabbing time data from the data list (which possibly
    contains multiple different packet types) one block at a time.
    """

    def __init__(self, data, track_info, block_size):
        self.__data = data
        self.__curr_index = 0
        self.__sample_index = 0
        self.__track_info = track_info
        self.__block_overhead = struct.calcsize('3Q')
        self.__packet_overhead = struct.calcsize('3Q')
        self.__nominal_size = struct.calcsize('L3d')
        self.__precision_size = struct.calcsize('LQ3dLQ')
        self.__block_size = block_size - self.__block_overhead

    def grab_block(self):
        packets = []
        remaining_bytes = self.__block_size
        total_size = self.__block_overhead
        item = None
        for self.__curr_index in xrange(self.__curr_index, len(self.__data)):
            item = self.__data[self.__curr_index]
            if isinstance(item, list):
                self.__sample_index += (len(item[0]) /
                                        self.__track_info[0].elems_per_sample)
            elif isinstance(item, dict) and ('Valid' in item):
                packet_size = 4 # for valid flag
                for stamp in item['TimeStamps']:
                    if stamp['RepresentsPTime']:
                        packet_size += self.__precision_size
                        if 'pi' in stamp:
                            packet_size += len(stamp['pi']) * 8
                    else:
                        packet_size += self.__nominal_size
                if packet_size + self.__packet_overhead <= remaining_bytes:
                    total_size += self.__packet_overhead + packet_size
                    packets.append((self.__sample_index, packet_size, item))
                else:
                    break

        # if the last item was a data item, increment the index so we
        # don't try to look at it again
        if isinstance(item, list):
            self.__curr_index += 1

        return packets, total_size

class EventDataGrabber:
    """
    Class for grabbing event data from the data list (which possibly
    contains multiple different packet types) one block at a time.
    """

    def __init__(self, data, track_info, event_track_info, block_size):
        self.__data = data
        self.__curr_index = 0
        self.__sample_index = 0
        self.__track_info = track_info
        self.__event_track_info = event_track_info
        self.__block_overhead = struct.calcsize('2d2Q')
        self.__track_overhead = struct.calcsize('2Q')
        self.__packet_overhead = struct.calcsize('2d2Q')
        self.__block_size = block_size - self.__block_overhead

    def grab_block(self):
        packets = []
        start_time = -1
        end_time = 0
        remaining_bytes = self.__block_size
        total_size = (self.__block_overhead +
                      len(self.__event_track_info) * self.__track_overhead)
        item = None
        for self.__curr_index in xrange(self.__curr_index, len(self.__data)):
            item = self.__data[self.__curr_index]
            if isinstance(item, list):
                self.__sample_index += (len(item[0]) /
                                        self.__track_info[0].elems_per_sample)
            elif isinstance(item, dict) and ('Valid' not in item):
                tracks = item['TRACKS']
                if len(tracks[0]) == 0:
                    packet_size = 0
                else:
                    packet_size = (len(tracks[0][0]) *
                                   self.__event_track_info[0].packet_size)
                if packet_size + self.__packet_overhead <= remaining_bytes:
                    total_size += self.__packet_overhead + packet_size
                    packets.append((self.__sample_index, packet_size, item))
                    if start_time == -1 or item['StartTime'] < start_time:
                        start_time = item['StartTime']
                    if item['EndTime'] > end_time:
                        end_time = item['EndTime']
                else:
                    break

        # if the last item was a data item, increment the index so we
        # don't try to look at it again
        if isinstance(item, list):
            self.__curr_index += 1

        return packets, total_size, start_time, end_time


# Match one or more contiguous comment lines, and all leading
# whitespace of the next line. Removes comments to end of line or end
# of string.
_comment_re = re.compile('^\s*(?://.*[\n\Z]\s*)*')

# Match a single identifier, cannot start with digits
# (?!R) matches if !R match and does not consume anything
_identifier_re = re.compile('[a-zA-Z_]\w*$')

# Match a single complex value (to be used in a sequence of them)
_vector_complex_re = re.compile('\(.*?\)')

# All following expressions below must return a single group and
# consume everything up to the next character to process (and no
# more).
# .*? is the nongreedy version of .*
# .* and .*? do not match across multiple lines, but \s* does
# (?=R) matches the regexp R, but does not consume it
_typed_re = re.compile('([^,}]*?\s*):\s*')

# A key can either by an identifier or a string literal, this match
# will produce 2 groups. One of them will be None.
_key_re = re.compile('(?:([a-zA-Z_]\w*)|"(.*?)(?<!\\\\)")\s*=\s*')
_value_regexps = {
  # The first group in each expression must be the targeted value string,
  # and consumes any trailing whitespace that follows (excluding comments)
  'vector': re.compile(  '(<.*?>)\s*'),
  'complex':re.compile('(\(.*?\))\s*'),
  # string consumes non-greedily to the first " that is not preceded by \
  # note that .* does not match \n so it is limited to a single line.
  'string': re.compile('"(.*?)(?<!\\\\)"\s*'),

  # value is tricky because unlike all of the expressions above, it is
  # not end delimited by a single character that we will be consuming.
  # Use a forward look-ahead match that doesn't consume to match
  # [\s,}] or a comment "//" THEN remove trailing whitespace (if any).
  # Note: this leaves the possibility of an empty string for the
  # matched value, which is not legal. Attempting to add this to the
  # regexpr conflicted with the look-ahead match. Hence, this is
  # checked for explicitly in the code that uses this regexpr.
  'value':  re.compile('(.*?)(?=[\s,}]|//)\s*')
}

def _match_type(vtype, data):
    match = _value_regexps[vtype].match(data)
    if not match:
        raise OpalFileException('Expected a %s in "%s"' % (vtype, data[:30]))
    return match.group(1), data[match.end():].lstrip()


_values_dict = {
    'UX': long,
    'UL': long,
    'UI': int,
    'UB': int,
    'X': long,
    'L': int,
    'I': int,
    'B': int,
    'D': float,
    'F': float,
    'DUR': float,
    'T': XMTime
    }

_type_to_numpy_dict = {
    'UX': N.dtype(N.uint64),
    'UL': N.dtype(N.uint32),
    'UI': N.dtype(N.uint16),
    'UB': N.dtype(N.uint8),
    'X': N.dtype(N.int64),
    'L': N.dtype(N.int32),
    'I': N.dtype(N.int16),
    'B': N.dtype(N.int8),
    'F': N.dtype(N.float32),
    'D': N.dtype(N.float64),
    'DUR': N.dtype(N.float64),
    'T': N.dtype(N.uint64),
    'BIT': ''
    }

_type_to_cnumpy_dict = {
    'UX': N.dtype(N.uint64),
    'UL': N.dtype(N.uint32),
    'UI': N.dtype(N.uint16),
    'UB': N.dtype(N.uint8),
    'X': N.dtype(N.int64),
    'L': N.dtype(N.int32),
    'I': N.dtype(N.int16),
    'B': N.dtype(N.int8),
    'F': N.dtype(N.complex64),
    'D': N.dtype(N.complex128),
    }

_numpy_to_m2k = {
    N.dtype(N.uint64): 'UX',
    N.dtype(N.uint32): 'UL',
    N.dtype(N.uint16): 'UI',
    N.dtype(N.uint8): 'UB',
    N.dtype(N.int64): 'X',
    N.dtype(N.int32): 'L',
    N.dtype(N.int16): 'I',
    N.dtype(N.int8): 'B',
    N.dtype(N.complex64): 'CF',
    N.dtype(N.complex128): 'CD',
    N.dtype(N.float32): 'F',
    N.dtype(N.float64): 'D'
    }

_m2k_size_dict = {
    'UX': 8,
    'X': 8,
    'UL': 4,
    'L': 4,
    'UI': 2,
    'I': 2,
    'UB': 1,
    'B': 1,
    'D': 8,
    'F': 4,
    'DUR': 8,
    'T': 8,
    'BIT': .125
    }

_numpy_native_complex = {
    'F': 1,
    'D': 1,
    'UX': 0,
    'X': 0,
    'UL': 0,
    'L': 0,
    'UI': 0,
    'I': 0,
    'UB': 0,
    'B': 0
    }

_dec_types = ['UX', 'X', 'UL', 'L']

# these simply make it easier to test
_long_long_m2k = 'X'
_long_long_sformat = 'q'
_long_long_usformat = 'Q'


def _read_complex(dtype, value):
    """
    Returns a Python complex.

    Expects a Midas2k type and a value in the form '(0, 0)'.
    """

    real, imag = value[1:-1].split(',', 1)
    real = _values_dict[dtype](real)
    imag = _values_dict[dtype](imag)
    return complex(real, imag)

def _read_dict_or_list(data):
    """
    Returns a dictionary or list and a string with the data remaining
    to be read.

    Expects the data stream to be pointing at the start of the
    OpalTable dictionary/array to read, i.e. data[0] == '{'
    """
    assert data[0] == '{'
    data = data[1:]
    ret_val = {}
    while True:
        if '//' in data:
            data = _comment_re.sub('', data)
        else:
            data = data.strip()
        if data[:1] == '}':
            data = data[1:]
            break

        # Extract key (if any)
        match = _key_re.match(data)
        if match:
            # An identifier or a string literal matched in the first 2 groups
            key = (match.group(1) or match.group(2).replace('\\"', '"'))
            data = data[match.end():]
        else:
            # The Midas2k parser supports { value, value, value ... }
            # generating string keys for them based on their position.
            key = str(len(ret_val))

        # Extract value
        if data[:1] == '"':
            # string
            value, data = _match_type('string', data)
            value = value.replace('\\"', '"')
        elif data[:1] == '{':
            # dictionary/list
            value, data = _read_dict_or_list(data)
        elif data[:1] == '@':
            # link to a dictionary/list
            value, data = _match_type('string', data[1:])
            value = value.replace('\\"', '"')
            value = Link(value)
        else:
            # typed value/typed vector/untyped
            value, data = _read_value(data)
        ret_val[key] = value

        # Legally, data should now start with "," or "}" or "//".
        # Leading whitespace was removed by the value extraction.
        if data[:1] == ',':
            data = data[1:]

    if '0' in ret_val:
        # Try making it into a Python list
        try:
            ret_val = [ret_val[str(x)] for x in xrange(len(ret_val))]
        except:
            pass

    return ret_val, data

def _read_vector(dtype, value_str):
    """
    Returns a vector as a python numpy array.

    Expects a Midas2k type and a value in the form '<0, 0, ...>'.
    Complex integer vectors are returned as 2 dimensional arrays.
    On platforms that don't support 8 byte integers, a numpy array
    of Python longs is returned.
    """

    value = value_str[1:-1]
    ret = []
    num_type = None
    if value:
        if dtype[0] != 'C':
            values = value.split(',')
            for val in values:
                ret.append(_values_dict[dtype](val.strip()))
            num_type = _type_to_numpy_dict[dtype]
        else:
            values = _vector_complex_re.findall(value)
            for val in values:
                ret.append(_read_complex(dtype[1:], val.strip()))
            num_type = _type_to_cnumpy_dict[dtype[1:]]

    return N.array(ret, num_type)

def _read_value(data):
    """
    Returns a value and a string with the data remaining
    to be read. The value could be any of the built-in python
    types or a numpy array.
    """
    match = _typed_re.match(data)
    if match:
        # typed value
        val_type = match.group(1)
        data = data[match.end():]
        if data[0] == '<':
           # vector
            value, data = _match_type('vector', data)
            value = _read_vector(val_type, value)
        elif data[0] == '(':
            # complex
            assert val_type[0] == 'C'
            value, data = _match_type('complex', data)
            value = _read_complex(val_type[1:], value)
        else:
            # value
            value, data = _match_type('value', data)
            value = _values_dict[val_type](value)
    else:
        # untyped: use default type
        value, data = _match_type('value', data)
        if not value:
            # no value found
            raise OpalFileException('Expected a value, encountered "%s"' %
                                    data[:30])
        try:
            value = _untyped_default(value)
        except:
            # default conversion failed, keep it as a string
            pass

    return value, data

def _read_block(infile, block_size, track_info, machine_rep, version,
                data, options, indices, order):
    """
    Reads a block of data from the file and appends each track block
    to the appropriate list in 'data'.
    """

    block = infile.read(block_size)
    block_index = 0
    temp_data = {}
    for track in xrange(0, len(track_info)):
        ti = track_info[track]
        # only read tracks that are actually requested
        if track in indices:
            vals = _read_array_data(buffer(block, block_index,
                                           ti.track_block_size),
                                    ti.m2k_format, ti.num_format,
                                    machine_rep, version, options)
            temp_data[track] = vals
        block_index += ti.track_block_size

    # now put them in the order they were requested
    assert len(order) == len(data)
    for ii in range(len(order)):
        data[ii].append(temp_data[order[ii]])

def _write_block(ofile, block, track_info, machine_rep, version):
    """
    Writes a block of data to 'ofile'. 'block' is a list of track blocks.
    """
    buf = cStringIO.StringIO()
    for track in xrange(0, len(block)):
        ti = track_info[track]
        _write_array_data(buf, block[track], ti.m2k_format, machine_rep,
                          version)

    if buf.tell():
        ofile.write(buf.getvalue())

def _create_key(key, k):
    if key is None:
        return str(k)
    return key + '.' + str(k)

def _write_array(ofile, arr, indent_level, dicts, key):
    """
    Writes 'arr' to 'ofile' in the output format 'UX:<1, 2, 3>'.
    """
    cplx = 0
    if N.rank(arr) == 2:
        ofile.write('C')
        cplx = 1

    # handle our special cases: longs and non-native complexes
    if arr.dtype == N.dtype(N.object_):
        if len(arr) > 0:
            if cplx:
                val = arr[0][0]
            else:
                val = arr[0]
            if isinstance(val, long):
                temp_arr = N.sort(arr.ravel())
                if temp_arr[0] < 0L:
                    if temp_arr[-1] > 9223372036854775807L:
                        raise OpalFileException('Invalid values in array')
                    else:
                        typestr = 'X'
                elif temp_arr[-1] <= 9223372036854775807L:
                    typestr = 'X'
                else:
                    typestr = 'UX'
            else:
                assert isinstance(val, XMTime)
                typestr = 'T'
        else:
            # assume 'UX' type if no values present
            typestr = 'UX'
    else:
        typestr = _numpy_to_m2k[arr.dtype]

    ofile.write(typestr)
    ofile.write(':<')

    if len(arr) > 0:
        if cplx:
            arr = arr.ravel()
            assert len(arr) % 2 == 0
            ofile.write('(')
            _write_val(ofile, arr[0], indent_level, dicts,
                       _create_key(key, 0), 0)
            ofile.write(', ')
            _write_val(ofile, arr[1], indent_level, dicts,
                       _create_key(key, 0), 0)
            ofile.write(')')
            for i in xrange(2, len(arr), 2):
                ofile.write(', (')
                _write_val(ofile, arr[i], indent_level, dicts,
                           _create_key(key, i), 0)
                ofile.write(', ')
                _write_val(ofile, arr[i+1], indent_level, dicts,
                           _create_key(key, i), 0)
                ofile.write(')')
        else:
            _write_val(ofile, arr[0], indent_level, dicts,
                       _create_key(key, 0), 0)
            for i in xrange(1, len(arr)):
                ofile.write(', ')
                _write_val(ofile, arr[i], indent_level, dicts,
                           _create_key(key, i), 0)

    ofile.write('>')

def _write_list(ofile, lst, indent_level, dicts, key, print_type):
    """
    Writes 'lst' to 'ofile'. Each entry in the list is written as
    a key/value pair. The key is the index in the list.
    E.g. [1,2] becomes {"0"=L:1,"1"=L:2}
    """
    # appease pychecker
    if print_type:
        pass

    ofile.write('{')
    if not lst:
        # NB: Unfortunately, this means that on the way back in,
        # this will be interpreted as an empty *dictionary*, not
        # an empty list. This isn't good, but is an unfortunate
        # consequence of M2k ASCII serialization:  empty lists and
        # empty OpalTables look the same.
        ofile.write(' }')
        return

    ofile.write(os.linesep)
    indent_level += 1
    ofile.write(_indent * indent_level)
    ofile.write('"0"=')
    _write_val(ofile, lst[0], indent_level, dicts, _create_key(key, 0))
    for i in xrange(1, len(lst)):
        ofile.write(',')
        ofile.write(os.linesep)
        ofile.write(_indent * indent_level)
        ofile.write('"')
        ofile.write(str(i))
        ofile.write('"=')
        _write_val(ofile, lst[i], indent_level, dicts, _create_key(key, i))
    indent_level -= 1
    ofile.write(os.linesep)
    ofile.write(_indent * indent_level)
    ofile.write('}')

def _write_string(ofile, string, indent_level, dicts, key, print_type):
    """
    Writes 'string' to 'ofile' surrounded by double quotes.
    """
    # appease pychecker
    if indent_level or dicts or key or print_type:
        pass

    ofile.write('"')
    ofile.write(string.replace('"', r'\"'))
    ofile.write('"')

def _write_long(ofile, l, indent_level, dicts, key, print_type):
    """
    Writes 'l' to 'ofile'. If 'print_type' is true, the M2K type
    followed by a colon is written first.
    """
    # appease pychecker
    if indent_level or dicts or key:
        pass

    if print_type:
        ofile.write(_dec_types[_get_long_type_index(l)])
        ofile.write(':')
    ofile.write(str(l))

def _write_int(ofile, i, indent_level, dicts, key, print_type):
    """
    Writes 'i' to 'ofile'. If 'print_type' is true, the M2K type
    followed by a colon is written first.
    """
    # appease pychecker
    if indent_level or dicts or key:
        pass

    if print_type:
        ofile.write(_dec_types[_get_int_type_index(i)])
        ofile.write(':')
    ofile.write(str(i))

def _write_float32(ofile, f, indent_level, dicts, key, print_type):
    """
    Writes 32 bit float value 'f' to 'file' (printed to 8 significant
    digits). If 'print_type' is true, the M2K type followed by a colon
    is written first.
    """
    # appease pychecker
    if indent_level or dicts or key:
        pass

    if print_type:
        ofile.write('F:')
    ofile.write('%.8g' % f)

def _write_float64(ofile, f, indent_level, dicts, key, print_type):
    """
    Writes 64 bit float value 'f' to 'file' (printed to 16 significant
    digits). If 'print_type' is true, the M2K type followed by a colon
    is written first.
    """
    # appease pychecker
    if indent_level or dicts or key:
        pass

    if print_type:
        ofile.write('D:')
    ofile.write('%.16g' % f)

_write_float = _write_float64

def _write_complex64(ofile, c, indent_level, dicts, key, print_type):
    """
    Writes 64 bit (total) complex value 'c' to 'file' (real and
    imaginary parts are printed to 8 significant digits). If 'print_type'
    is true, the M2K type followed by a colon is written first.
    """
    # appease pychecker
    if indent_level or dicts or key:
        pass

    if print_type:
        ofile.write('CF:')
    ofile.write('(%.8g, %.8g)' % (c.real, c.imag))

def _write_complex128(ofile, c, indent_level, dicts, key, print_type):
    """
    Writes 128 bit (total) complex value 'c' to 'file' (real and
    imaginary parts are printed to 16 significant digits). If 'print_type'
    is true, the M2K type followed by a colon is written first.
    """
    # appease pychecker
    if indent_level or dicts or key:
        pass

    if print_type:
        ofile.write('CD:')
    ofile.write('(%.16g, %.16g)' % (c.real, c.imag))

_write_complex = _write_complex128

def _get_long_type_index(l):
    """
    Returns the index into the types array corresponding to this long
    """
    if l > 9223372036854775807L:
        return 0
    elif l > 4294967295L or l < -2147483648L:
        return 1
    elif l > 2147483647L:
        return 2
    else:
        return 3

def _get_int_type_index(i):
    """
    Returns the index into the types array corresponding to this int
    """
    if i > 2147483647:
        return 2
    else:
        return 3

# complexes that can't be represented with python complex
def _write_tuple(ofile, t, indent_level, dicts, key, print_type):
    """
    Writes a tuple to 'ofile' as a M2K complex. If 'print_type' is
    true, the M2K type followed by a colon is written first.
    """
    # appease pychecker
    if indent_level or dicts or key:
        pass

    if print_type:
        ofile.write('C')

        if isinstance(t[0], int):
            func = _get_int_type_index
        else:
            func = _get_long_type_index

        type0 = func(t[0])
        type1 = func(t[1])
        if type0 < type1:
            ofile.write(_dec_types[type0])
        else:
            ofile.write(_dec_types[type1])

        ofile.write(':')

    ofile.write('(%s, %s)' % (t[0], t[1]))


def _write_xmtime(ofile, xt, indent_level, dicts, key, print_type):
    """
    Writes 'xt' to 'ofile'. If 'print_type' is true, the M2K type
    followed by a colon is written first.
    """

    # appease pychecker
    if indent_level or dicts or key:
        pass

    if print_type:
        ofile.write('T:')
    ofile.write(str(xt))

def _write_val(ofile, val, indent_level, dicts, key, print_type = 1):
    """
    Writes any value to 'ofile'. If 'print_type' is true, the M2K type
    followed by a colon is written first.
    """
    conv_method = None
    if isinstance(val, N.ndarray):
        _write_array(ofile, val, indent_level, dicts, key)
        return
    elif isinstance(val, N.number):
        # numpy scalar type. These are returned when indexing a
        # numpy array whose internal data type is not a native
        # Python type. First look explicitly for the float32 types, as
        # the default precision of the closest native Python numpy
        # type ('float') is inappropriate
        if val.dtype == N.dtype(N.float32):
            conv_method = _write_float32
        elif val.dtype == N.dtype(N.complex64):
            conv_method = _write_complex64
        else:
            # Use the type of the internal value, calling val.item()
            # forces it to be converted to a Python type
            val = val.item()

    if conv_method is None:
        clss = val.__class__
        conv_method = (_write_types.get(clss) or
                       _write_types_foreign.get(str(clss)))

    # No direct match; try serial polymorphic check
    if conv_method is None:
        for the_type, the_func in _write_types_list:
            if isinstance(val, the_type):
                conv_method = the_func
                break
        else:
            raise OpalFileException('Do not know how to convert key "%s" '
                                    'of type %s to an ASCII file' %
                                    (key, str(type(val))))
    conv_method(ofile, val, indent_level, dicts, key, print_type)

def _write_dict(ofile, d, indent_level, dicts, key, unused_print_type):
    """
    Writes a dictionary to 'ofile'. Recurses so nested elements will
    be written properly. If 'd' has already been written to the file,
    a M2K link will be written instead of the actual dictionary.
    """
    for di in dicts:
        if id(di[0]) == id(d):
            ofile.write('@"')
            ofile.write(di[1])
            ofile.write('"')
            return
    if d:
        dicts.append((d, key))

    ofile.write('{')

    def fix_key(key):
        if key[0] != '"' and _identifier_re.match(key) is None:
            key = '"' + key + '"'
        return key

    def cmpr(x, y):
        if x[0] == '"':
            cmp1 = x[1:-1]
        else:
            cmp1 = x
        if y[0] == '"':
            cmp2 = y[1:-1]
        else:
            cmp2 = y
        if cmp1 == cmp2:
            return 0
        elif cmp1 > cmp2:
            return 1
        else:
            return -1

    keys = d.keys()
    if keys:
        keys.sort(cmpr)
        indent_level += 1
        ofile.write(os.linesep)
        ofile.write(_indent * indent_level)
        ofile.write(fix_key(keys[0]))
        ofile.write('=')
        _write_val(ofile, d[keys[0]], indent_level, dicts,
                   _create_key(key, keys[0]))

        for i in xrange(1, len(keys)):
            k = keys[i]
            ofile.write(',')
            ofile.write(os.linesep)
            ofile.write(_indent * indent_level)
            ofile.write(fix_key(k))
            ofile.write('=')
            _write_val(ofile, d[k], indent_level, dicts, _create_key(key, k))

        ofile.write(os.linesep)
        indent_level -= 1
        ofile.write(_indent * indent_level)
    else:
        ofile.write(' ')

    ofile.write('}')

_write_types_list = [
    (dict, _write_dict),
    (list, _write_list),
    (str, _write_string),
    (long, _write_long),
    (int, _write_int),
    (float, _write_float),
    (complex, _write_complex),
    (tuple, _write_list),
    (XMTime, _write_xmtime),
    ]

_write_types_foreign = {
    # _write_xmtime prepends a T: and just calls str() on the value.
    # This ends up working for native M2Time classes as well.
    # We use the string representation of the type here because if
    # the initial look up in this table fails, we stringize the class
    # and try again. Thus we don't HAVE to have M2k support loaded
    # at the top level of this module.
    "<class 'm2time.M2Time'>": _write_xmtime
    }

_write_types = dict(_write_types_list)

def _open_file(name, mode):
    """
    Opens a file in the specified mode. If the mite or M2kinter module
    is available the path given is not absolute, the writepath or
    datapath (depending on the specified mode) is searched first.
    """
    if not name:
        raise OpalFileException('File name cannot be empty')

    name = os.path.expanduser(os.path.expandvars(name))

    infile = None
    if name[0] != os.sep:
        try:
            import mite as m2
        except:
            try:
                import M2kinter as m2
            except:
                m2 = None

        if m2:
            if mode[0] == 'w':
                paths = m2.writepath()
            else:
                assert mode[0] == 'r'
                paths = m2.datapath()

            for path in paths:
                if path:
                    if path[-1] != os.sep:
                        path += os.sep
                    if os.path.isfile(path + name) or mode[0] == 'w':
                        infile = open(path + name, mode)
                        if infile:
                            break
    if not infile and (os.path.isfile(name) or mode[0] == 'w'):
        infile = open(name, mode)

    return infile

def writetable(file_name, tbl):
    """
    Returns the path to the written file as a string.

    Writes the python dictionary, 'tbl', to a file in M2K OPAL
    format. If the file name given is not an absoulte path and the
    mite or M2kinter module is available, the writepath will be used
    for the file location.
    """

    buf = cStringIO.StringIO()
    _write_dict(buf, tbl, 0, [], None, None)
    infile = _open_file(file_name, 'wb')
    path = infile.name
    infile.write(buf.getvalue())
    infile.write(os.linesep)
    infile.flush()
    infile.close()
    return path

def writeheader(base_name, hdr):
    """
    Returns the path to the written file as a string.

    Writes the python dictionary, 'hdr', to a file in M2K OPAL
    format. '.hdr' will be appended to 'base_name' to get the final
    file name to write to. If the base name given is not an absolute
    path and the mite or M2kinter module is available, the writepath
    will be used for the file location.
    """

    base_name = _add_extension(base_name, '.hdr')
    return writetable(base_name, hdr)

def _verify_data(data, total_samples, track_info):
    """
    Throws an exception if the data doesn't match what the header says
    it should contain. Checks data type and length.
    """

    elems = []
    for ti in track_info:
        elems.append(0)
    for item in data:
        if isinstance(item, list):
            if len(item) != len(track_info):
                msg = 'Number of tracks in header (%i) does ' % len(track_info)
                msg += 'not match number of tracks in data (%i)' % len(item)
                raise OpalFileException(msg)
            for i in xrange(0, len(track_info)):
                info = track_info[i]
                arr = item[i]
                if isinstance(arr, basestring):
                    elems[i] += len(arr) * 8
                else:
                    elems[i] += len(arr.ravel())

                    # check if types match
                    if info.m2k_format[0] == 'C':
                        fmt = info.m2k_format[1:]
                        if (not _numpy_native_complex[fmt] and
                            N.rank(arr) != 2):
                            raise OpalFileException('Track %s holds ' +
                                                    'non-native complexes, ' +
                                                    'but is not dimension 2')
                        track_format = _type_to_cnumpy_dict[fmt]
                    else:
                        track_format = _type_to_numpy_dict[info.m2k_format]
                    if arr.dtype != track_format:
                        raise OpalFileException("Track %s is '%s' type, it "
                                                "should be '%s' type." %
                                                (i, arr.dtype, track_format))

    # check if have proper number of elements
    for i in xrange(0, len(track_info)):
        ti = track_info[i]
        num_elems = ti.elems_per_sample * total_samples
        if num_elems != elems[i]:
            raise OpalFileException('Track %s has %s elements, ' %
                                    (i, elems[i]) +
                                    'it should have %s elements.' % num_elems)

def writedata(base_name, hdr, data):
    """
    Returns the path to the written file as a string.

    Writes the continuous data one block at a time to a file in M2K
    OPAL format. '.data' will be appended to 'base_name' to get the
    final file name to write to. If base_name given is not an absolute
    path and the mite or M2kinter module is available, the writepath
    will be used for the file location.
    """

    base_name = _add_extension(base_name, '.data')
    path = None

    if data:
        ofile = _open_file(base_name, 'wb')
        path = ofile.name

        tracks = hdr['TRACKS']
        total_samples = hdr['TIME']['LENGTH']
        machine_rep = hdr['MACHINE_REP']
        granularity = hdr['GRANULARITY']
        if 'FILE_VERSION' not in hdr:
            file_version = 0
        else:
            file_version = int(float(hdr['FILE_VERSION']))

        track_info = get_track_info(tracks, granularity, total_samples)[0]

        _verify_data(data, total_samples, track_info)

        data_grabber = DataGrabber(data, track_info, granularity)
        i = -1
        remaining_samples = total_samples

        # by doing the range up to 'total_samples-granularity', we
        # guarantee we won't try to write a partial block here
        for i in xrange(0, total_samples-granularity, granularity):
            _write_block(ofile, data_grabber.grab_block(),
                         track_info, machine_rep, file_version)
            remaining_samples -= granularity

        if i == -1:
            # the file consists of either 1 full block, or 1 partial block
            i = 0
        else:
            # the file consists of at least 1 full block
            i += granularity

        # if total_samples is less than granularity, we have a partial block
        # at the end of the file, recalculate the track and block sizes
        if remaining_samples < granularity:
            track_info = get_track_info(tracks, granularity, total_samples,
                                        remaining_samples, track_info)[0]

        _write_block(ofile, data_grabber.grab_block(),
                     track_info, machine_rep, file_version)

        ofile.flush()
        ofile.close()
    return path

def _resolve_links(tbl, node):
    """
    Resolves references to Link objects to point to the actual object being
    referred to.
    """
    if isinstance(node, dict):
        for key in node.keys():
            _resolve_link(tbl, node, key)
    elif isinstance(node, list):
        for i in xrange(0, len(node)):
            _resolve_link(tbl, node, i)

def _resolve_link(tbl, node, key):
    if isinstance(node[key], Link):
        links = node[key].path.split('.')
        d = tbl
        try:
            for link in links:
                if isinstance(d, list):
                    link = int(link)
                d = d[link]
        except KeyError:
            raise OpalFileException('Invalid link: ' + node[key].path)
        except ValueError:
            raise OpalFileException('Invalid link: ' + node[key].path)
        node[key] = d
        # check if we just resolved to another link
        if isinstance(node[key], Link):
            _resolve_link(tbl, node, key)
    else :
        _resolve_links(tbl, node[key])

def readtable(file_name):
    """
    Returns the OPAL table file as a Python dictionary or list (depending
    on whether the top level Opal table is a hash or an OpalTable Array).
    """
    infile = _open_file(file_name, 'r')
    if not infile:
        raise OpalFileException('File ' + file_name + ' not found.')
    # Remove any leading comments from the top
    data = infile.read()
    if '//' in data:
        data = _comment_re.sub('', data)
    else:
        data = data.strip()
    infile.close()
    if data:
        retval = _read_dict_or_list(data)[0]
        _resolve_links(retval, retval)
    else:
        retval = None
    return retval

def readheader(base_name):
    """
    Returns a dictionary.

    Returns the OPAL header file as a Python dictionary. If 'base_name'
    does not have the '.hdr' extension, it is added before reading the
    file.
    """

    base_name = _add_extension(base_name, '.hdr')
    return readtable(base_name)

def readdata(base_name, hdr, which_tracks=None, options=0):
    """
    Returns a list of a list of numpy arrays.

    Unpacks the continuous data from the .data file into a list of a list
    of numpy arrays where each array corresponds to a track.
    """

    if which_tracks is None:
        which_tracks = []

    data = []

    base_name = _add_extension(base_name, '.data')
    infile = _open_file(base_name, 'rb')
    if not infile:
        return data

    tracks = hdr['TRACKS']
    if tracks:
        total_samples = hdr['TIME']['LENGTH']
        machine_rep = hdr['MACHINE_REP']
        granularity = hdr['GRANULARITY']
        if 'FILE_VERSION' not in hdr:
            file_version = 0
        else:
            file_version = int(float(hdr['FILE_VERSION']))

        track_info, block_size, track_names = get_track_info(tracks,
                                                             granularity,
                                                             total_samples)

        indices, order = _get_track_indices(which_tracks, track_info,
                                            track_names)
        for _ in order:
            data.append([])

        i = -1

        remaining_samples = total_samples

        # by doing the range up to 'total_samples-granularity', we
        # guarantee we won't try to read a partial block here
        for i in xrange(0, total_samples-granularity, granularity):
            _read_block(infile, block_size, track_info, machine_rep,
                        file_version, data, options, indices, order)
            remaining_samples -= granularity

        if i == -1:
            # the file consists of either 1 full block, or 1 partial block
            i = 0
        else:
            # the file consists of at least 1 full block
            i += granularity

        # if total_samples is less than granularity, we have a partial block
        # at the end of the file, recalculate the track and block sizes
        if remaining_samples < granularity:
            track_info, block_size, track_names = get_track_info(tracks,
                                                                 granularity,
                                                                 total_samples,
                                                                 remaining_samples,
                                                                 track_info)

        _read_block(infile, block_size, track_info, machine_rep, file_version,
                    data, options, indices, order)

        for ii in range(len(order)):
            if isinstance(data[ii][0], basestring):
                buf = cStringIO.StringIO()
                for s in data[ii]:
                    buf.write(s)
                data[ii] = buf.getvalue()
            else:
                data[ii] = N.concatenate(data[ii])
            tracks[order[ii]]['FORMAT'] = _convert_format(options, tracks[order[ii]]['FORMAT'])
            if len(track_info[order[ii]].dimensions) > 1:
                data[ii].shape = track_info[order[ii]].dimensions

    infile.close()
    return [data]

def get_track_info(tracks, granularity, total_samples=None,
                   block_samples=None, track_info=None):
    """
    Returns a list of TrackInfo objects and the total block size.

    Computes per track statistics and total block size for use when
    reading and writing tracks.
    """

    track_names = {}
    block_size = 0
    if track_info:
        # we've already computed statistics, just compute the new sizes
        # based on the number of samples given
        for t in xrange(0, len(tracks)):
            track = tracks[t]
            ti = track_info[t]
            track_block_size = long(ti.sample_size * block_samples)
            if ti.m2k_format == 'BIT':
                # round up to nearest byte
                track_block_size += (track_block_size % 8)
            ti.track_block_size = track_block_size
            ti.elems_per_block = ti.elems_per_sample * block_samples
            ti.samples_per_block = block_samples
            block_size += track_block_size
            track_names[ti.name] = t
    else:
        track_info = []
        for t in xrange(0, len(tracks)):
            track = tracks[t]

            # calculate the number of elements per sample
            if 'AXES' not in track:
                raise OpalFileException('AXES field of track ' + str(t) +
                                        ' is missing')
            axes = track['AXES']
            dimensions = []
            if 'LENGTH' not in axes[0]:
                raise OpalFileException('LENGTH field of track ' + str(t) +
                                        ' of axis 0 is missing')
            dimensions.append(axes[0]['LENGTH'])
            elems_per_sample = 1
            for i in xrange(1, len(axes)):
                if 'LENGTH' not in axes[i]:
                    raise OpalFileException('LENGTH field of track ' + str(t) +
                                            ' of axis ' + str(i) + ' is missing')
                elems_per_sample *= axes[i]['LENGTH']
                dimensions.append(axes[i]['LENGTH'])

            if 'FORMAT' not in track:
                raise OpalFileException('FORMAT field of track ' + str(t) +
                                        ' is missing')
            m2k_format = track['FORMAT']

            if m2k_format[0] == 'C':
                fmt = m2k_format[1:]
                # complex types are twice as large as normal types
                sample_size = elems_per_sample * _m2k_size_dict[fmt] * 2
                # we don't have native support in N.array for anything
                # but doubles and floats. we represent non native types as
                # a 2 dimension array so we have twice the number of elements
                if not _numpy_native_complex[fmt]:
                    elems_per_sample *= 2
                    num_format = _type_to_cnumpy_dict[fmt]
                else:
                    num_format = _type_to_cnumpy_dict[fmt]
            else:
                sample_size = elems_per_sample * _m2k_size_dict[m2k_format]
                num_format = _type_to_numpy_dict[m2k_format]

            # the number of bytes per track per block
            track_block_size = long(sample_size * granularity)

            if m2k_format == 'BIT':
                if track_block_size % 8:
                    raise OpalFileException('Track block size must be ' +
                                            'divisible by 8 for BIT data')

            # store the calculated info for use when reading
            ti = TrackInfo()
            ti.elems_per_sample = elems_per_sample
            ti.elems_per_block = elems_per_sample * granularity
            if total_samples is not None:
                ti.total_elements = elems_per_sample * total_samples
            ti.m2k_format = m2k_format
            ti.num_format = num_format
            ti.sample_size = sample_size
            ti.track_block_size = track_block_size
            ti.samples_per_block = granularity
            ti.dimensions = dimensions
            ti.name = track['NAME']
            track_info.append(ti)

            track_names[ti.name] = t
            block_size += track_block_size
    return track_info, block_size, track_names

def _add_extension(base_name, ext):
    """
    Returns a string.

    If the string doesn't already have the extension, add it.
    """
    base_name = os.path.splitext(base_name)[0]
    if base_name.endswith(os.sep):
        base_name = base_name[-1]
    base_name += ext
    return base_name

def writetime(base_name, hdr, data):
    """
    Returns the path to the written file as a string.
    """

    base_name = _add_extension(base_name, '.time')
    path = None

    if data:
        tracks = hdr['TRACKS']
        total_samples = hdr['TIME']['LENGTH']
        machine_rep = hdr['MACHINE_REP']
        granularity = hdr['GRANULARITY']

        if 'FILE_VERSION' not in hdr:
            file_version = 0
        else:
            file_version = int(float(hdr['FILE_VERSION']))

        track_info, block_size = get_track_info(tracks, granularity,
                                                total_samples)[:2]

        data_grabber = TimeDataGrabber(data, track_info, block_size)
        packets, size = data_grabber.grab_block()
        if packets:
            if file_version < 2:
                raise OpalFileException('Time file version ' +
                                        str(file_version) + ' not supported')
            ofile = _open_file(base_name, 'wb')
            path = file.name

            while packets:
                _write_time_block(ofile, packets, size, machine_rep,
                                  file_version)
                packets, size = data_grabber.grab_block()

            # write dummy header to indicate end
            bh = struct.pack('%s3Q' % _struct_endian[machine_rep], 0, 0, 0)
            ofile.write(bh)

            ofile.flush()
            ofile.close()
    return path

def _write_time_block(ofile, packets, size, rep, version):
    """
    Writes a block of time packets to 'ofile'.

    It is assumed that packets is a list of tuples where each tuple
    is the following:
    (sample_index, packet_size, packet)
    It is also assumed that packets contains exactly the number of
    packets that should be written out for this block. 'size' is the
    total size of the block.
    """
    offset = ofile.tell()
    buf = cStringIO.StringIO()
    bh_fmt = '3Q'
    bh = struct.pack('%s%s' % (_struct_endian[rep], bh_fmt), offset,
                     offset + size, len(packets))
    buf.write(bh)
    offset += struct.calcsize(bh_fmt)

    ph_fmt = '3Q'
    ph_size = struct.calcsize(ph_fmt)
    offset += ph_size * len(packets)
    for packet_tuple in packets:
        ph = struct.pack('%s%s' % (_struct_endian[rep], ph_fmt),
                         0,
                         packet_tuple[0],
                         offset)
        buf.write(ph)
        offset += packet_tuple[1]

    if version < 3:
        nominal_fmt = 'LQ2d'
    else:
        nominal_fmt = 'L3d'
    precision_fmt = 'LQ3dLQ'
    for packet_tuple in packets:
        packet = packet_tuple[2]
        valid = struct.pack('%sL' % _struct_endian[rep], packet['Valid'])
        buf.write(valid)
        for stamp in packet['TimeStamps']:
            fso = stamp['FSO'] - packet_tuple[0]
            vso = stamp['VSO'] - packet_tuple[0]
            if stamp['RepresentsPTime']:
                integral, fractional = stamp['Time'].two_floats()
                if version < 3:
                    time = long(integral * 1e9) + long(fractional * 1e9)
                else:
                    time = long(integral * 4e9) + long(fractional * 4e9)
                if 'pi' in stamp:
                    num_coef = len(stamp['pi'])
                else:
                    num_coef = 0
                body = struct.pack('%s%s' % (_struct_endian[rep],
                                             precision_fmt),
                                   1, time, fso, vso,
                                   stamp['AccumDecimation'],
                                   stamp['TrackNumber'], num_coef)
                buf.write(body)
                if num_coef:
                    coef = struct.pack('%s%sd' % (_struct_endian[rep],
                                                  num_coef),
                                       *stamp['pi'])
                    buf.write(coef)
            else:
                time = stamp['Time']
                if version < 3:
                    time *= 1e9
                body = struct.pack('%s%s' % (_struct_endian[rep],
                                             nominal_fmt),
                                   0, time, fso, vso)
                buf.write(body)
    ofile.write(buf.getvalue())

def _read_time_block(ofile, offset, rep, version, packets):
    """
    Returns a long.

    The return value is the offset in the file of the next time
    block header. All read time packets are added to the 'packets'
    dictionary.
    """

    if version > 1:
        tmh_format = '3Q'
    else:
        tmh_format = '2Q'
    size = struct.calcsize(tmh_format)
    ofile.seek(offset)
    string = ofile.read(size)
    vals = struct.unpack('%s%s' % (_struct_endian[rep], tmh_format), string)
    if offset != vals[0]:
        # dummy header to indicate end of headers
        assert vals[0] == 0
        return -1
    next_offset = vals[1]
    block_size = next_offset - offset
    ofile.seek(offset)
    block = ofile.read(block_size)
    curr = size

    packet_list = []

    # read in all the packet headers first
    tph_format = '3Q'
    tph_size = struct.calcsize(tph_format)
    if version > 1:
        num_packets = vals[2]
        for i in xrange(0, num_packets):
            temp = buffer(block, curr, tph_size)
            vals = struct.unpack('%s%s' %
                                 (_struct_endian[rep],tph_format), temp)
            packet = {'SampleIndex': vals[1], 'Offset': vals[2]}
            packet_list.append(packet)
            _append_packet(packets, vals[1], packet)
            curr += tph_size
    else:
        temp = buffer(block, curr, tph_size)
        vals = struct.unpack('%s%s' % (_struct_endian[rep], tph_format), temp)
        stop = vals[2]
        packet = {'SampleIndex': vals[1], 'Offset': vals[2]}
        packet_list.append(packet)
        _append_packet(packets, vals[1], packet)
        curr += tph_size
        while curr < stop:
            temp = buffer(block, curr, tph_size)
            vals = struct.unpack('%s%s' %
                                 (_struct_endian[rep],tph_format), temp)
            packet = {'SampleIndex': vals[1], 'Offset': vals[2]}
            packet_list.append(packet)
            _append_packet(packets, vals[1], packet)
            curr += tph_size

    # now use the header data to read in the packet bodies
    if curr > size:
        for i in xrange(0, len(packet_list)-1):
            packet = packet_list[i]
            end = packet_list[i+1]['Offset'] - offset
            _read_time_packet(packet, buffer(block, curr, end-curr),
                              rep, version)
            del packet['Offset']
            del packet['SampleIndex']
            curr = end
        _read_time_packet(packet_list[len(packet_list)-1],
                          buffer(block, curr, next_offset-curr), rep, version)
        del packet_list[len(packet_list)-1]['Offset']
        del packet_list[len(packet_list)-1]['SampleIndex']

    return next_offset

def _append_packet(packets, index, packet):
    """
    Adds a packet to the packets dictionary.
    """
    if index not in packets:
        packets[index] = []
    packets[index].append(packet)

def _read_time_packet(packet, string, rep, version):
    """
    Unpacks a single time packet from 'string' given the time packet
    header 'packet'. All packet fields and samples will be added
    to the 'packet' dictionary.
    """

    packet['Valid'] = struct.unpack('%sl' % _struct_endian[rep],
                                    buffer(string, 0, 4))[0]
    samples = []

    curr = 4
    while curr < len(string):
        sample, num_read = _read_time_sample(buffer(string, curr),
                                             packet['SampleIndex'],
                                             rep,
                                             version)
        curr += num_read
        samples.append(sample)

    packet['TimeStamps'] = samples

def _read_time_sample(string, sample_index, rep, version):
    """
    Returns a dictionary and the number of bytes read.

    'sample_index' is used to adjust the FSO and VSO of each sample.
    """

    sample = {}
    sample['RepresentsPTime'] = struct.unpack('%sl' % _struct_endian[rep],
                                              buffer(string, 0, 4))[0]
    curr = 4
    if sample['RepresentsPTime']:
        fmt = 'Q3dLQ'
        size = struct.calcsize(fmt)
        vals = struct.unpack('%s%s' %  (_struct_endian[rep], fmt),
                             buffer(string, curr, size))
        curr += size
        sample['Time'] = _convert_to_XMTime(vals[0], (version > 2))
        sample['AccumDecimation'] = vals[3]
        sample['TrackNumber'] = vals[4]

        num_coefficients = vals[5]
        if num_coefficients:
            fmt = 'd'
            nbytes = struct.calcsize(fmt) * num_coefficients
            vals = struct.unpack('%s%s%s' % (_struct_endian[rep],
                                            num_coefficients,
                                            fmt),
                                 buffer(string, curr, nbytes))
            sample['pi'] = N.float64(vals)
            curr += nbytes
    else:
        if version < 3:
            fmt = 'Q2d'
        else:
            fmt = '3d'
        size = struct.calcsize(fmt)
        vals = struct.unpack('%s%s' %  (_struct_endian[rep], fmt),
                             buffer(string, curr, size))
        curr += size
        sample['Time'] = vals[0]
        if version < 3:
            sample['Time'] = _convert_to_real8(sample['Time'])

    sample['FSO'] = vals[1] + sample_index
    sample['VSO'] = vals[2] + sample_index

    return sample, curr

def _convert_to_integral_fractional(uint8, is_qns):
    """
    Returns two floats.

    Converts an 8 byte integer to integral and fractional floats.
    """
    if not is_qns:
        fractional = uint8 % 1000000000L
        integral = uint8 - fractional
        integral *= 1e-9
        fractional *= 1e-9
    else:
        fractional = uint8 % 4000000000L
        integral = uint8 - fractional
        integral *= .25e-9
        fractional *= .25e-9
    return integral, fractional

def _convert_to_XMTime(uint8, is_qns=1):
    """
    Returns a XMTime.

    Converts an 8 byte integer to a XMTime with as much precision as possible.
    """
    integral, fractional = _convert_to_integral_fractional(uint8, is_qns)
    return XMTime(integral, fractional)

def _convert_to_real8(uint8, is_qns=0):
    """
    Returns a float.

    Converts an 8 byte integer to a real8 with as much precision as possible.
    """
    integral, fractional = _convert_to_integral_fractional(uint8, is_qns)
    return integral + fractional

def _convert_to_uint8(real8, is_qns=1):
    """
    Returns a long.

    Converts an 8 byte real to a long with as much precision as possible.
    """
    if not isinstance(real8, tuple):
        t = XMTime(real8)
        integral, fractional = t.two_floats()
    else:
        integral, fractional = real8
    if not is_qns:
        integral = long(integral) * 1000000000L
        fractional *= 1000000000L
    else:
        integral = long(integral) * 4000000000L
        fractional *= 4000000000L
    return integral + long(fractional)

def _read_time(base_name, hdr):
    """
    Returns a dictionary.

    Reads the .time file and returns all time packets in a dictionary.
    Each key in the dictionary is a sample index into the continuous
    data. The value for each key is a list of time packets.
    """

    base_name = _add_extension(base_name, '.time')
    infile = _open_file(base_name, 'rb')
    if not infile:
        return {}

    machine_rep = hdr['MACHINE_REP']

    if 'FILE_VERSION' not in hdr:
        file_version = 0
    else:
        file_version = int(float(hdr['FILE_VERSION']))

    if file_version < 2:
        raise OpalFileException('Time file version ' + str(file_version) +
                                ' not supported')

    packets = {}
    next_offset = _read_time_block(infile, 0, machine_rep,
                                   file_version, packets)
    while (next_offset != -1):
        next_offset = _read_time_block(infile, next_offset, machine_rep,
                                       file_version, packets)

    infile.close()
    return packets

def readtime(base_name, hdr):
    """
    Returns a list of packets.

    Reads the .time file and returns all time packets in a single list.
    """
    packets = _read_time(base_name, hdr)
    keys = packets.keys()
    keys.sort()
    ret = []
    for key in keys:
        ret.extend(packets[key])
    return ret

def _write_event_block(ofile, packets, track_info, start_time, end_time, size,
                       rep, version, is_last, time_delta):
    """
    Writes a block of event data to 'ofile'.

    'size' is the total size of the block.
    """
    offset = ofile.tell()
    buf = cStringIO.StringIO()

    num_packets = len(packets)

    if version < 3:
        bh_fmt = '4Q'
        start_time = _convert_to_uint8(start_time, 0)
        end_time = _convert_to_uint8(end_time, 0)
    else:
        bh_fmt = '2d2Q'
    if is_last:
        next_offset = 0
    else:
        next_offset = offset + size
    bh = struct.pack('%s%s' % (_struct_endian[rep], bh_fmt), start_time,
                     end_time, num_packets, next_offset)
    buf.write(bh)

    num_tracks = len(track_info)

    # initialize our tracks structure
    tracks = []
    track_sizes = []
    for i in xrange(0, num_tracks):
        num_fields = len(packets[0][2]['TRACKS'][i])
        track_sizes.append([0, 0]) # num events, size in bytes
        tracks.append([])
        for j in xrange(0, num_fields):
            tracks[i].append([])

    # concatenate field data
    for packet_tuple in packets:
        packet = packet_tuple[2]
        for i in xrange(0, num_tracks):
            num_fields = len(packet['TRACKS'][i])
            for j in xrange(0, num_fields):
                tracks[i][j].append(packet['TRACKS'][i][j])

    # turn list of field data into a single array
    for i in xrange(0, num_tracks):
        track_sizes[i][0] += (len(tracks[i][0]) /
                              track_info[i].fields[0].elems_per_sample)
        num_fields = len(packet['TRACKS'][i])
        for j in xrange(0, num_fields):
            if isinstance(tracks[i][j][0], basestring):
                buf = cStringIO.StringIO()
                for s in tracks[i][j]:
                    buf.write(s)
                tracks[i][j] = buf.getvalue()
            else:
                tracks[i][j] = N.concatenate(tracks[i][j])
            track_sizes[i][1] += (track_sizes[i][0] *
                                  track_info[i].packet_size)

    th_fmt = '2Q'
    if version < 3:
        ph_fmt = '4Q'
    else:
        ph_fmt = '2d2Q'
    track_offset = (offset + buf.tell() + len(tracks) *
                    struct.calcsize(th_fmt) +
                    num_packets * struct.calcsize(ph_fmt))
    for i in xrange(0, len(tracks)):
        track = tracks[i]
        ti = track_info[i]
        th = struct.pack('%s%s' % (_struct_endian[rep], th_fmt),
                         track_sizes[i][0], track_offset)
        buf.write(th)
        track_offset += track_sizes[i][1]

    total_samples = long(round((end_time - start_time) / time_delta))

    if version < 3:
        for packet_tuple in packets:
            packet = packet_tuple[2]
            samples_offset = long(round((packet['StartTime'] - start_time) /
                                  time_delta))
            if packet['EndTime'] <= packet['StartTime']:
                continuous = total_samples - samples_offset
            else:
                continuous = long(round((packet['EndTime'] -
                                         packet['StartTime']) /
                                  time_delta))
            ph = struct.pack('%s%s' % (_struct_endian[rep], ph_fmt),
                             _convert_to_uint8(packet['StartTime'], 0),
                             _convert_to_uint8(packet['EndTime'], 0),
                             continuous,
                             packet_tuple[0])
            buf.write(ph)
    else:
        for packet_tuple in packets:
            packet = packet_tuple[2]
            ph = struct.pack('%s%s' % (_struct_endian[rep], ph_fmt[:2]),
                             packet['StartTime'], packet['EndTime'])
            buf.write(ph)

        for packet_tuple in packets:
            packet = packet_tuple[2]
            samples_offset = long(round((packet['StartTime'] - start_time) /
                                  time_delta))
            if packet['EndTime'] <= packet['StartTime']:
                continuous = total_samples - samples_offset
            else:
                continuous = long(round((packet['EndTime'] -
                                        packet['StartTime']) /
                                  time_delta))
            ph = struct.pack('%s%s' % (_struct_endian[rep], ph_fmt[2:]),
                             continuous, packet_tuple[0])
            buf.write(ph)

    for i in xrange(0, len(tracks)):
        track = tracks[i]
        ti = track_info[i]
        for j in xrange(0, len(track)):
            field = track[j]
            _write_array_data(buf, field, ti.fields[j].m2k_format,
                              rep, version)

    ofile.write(buf.getvalue())


def writeevent(base_name, hdr, data):
    """
    Returns the path to the written file as a string.
    """

    base_name = _add_extension(base_name, '.evt')
    path = None

    if data and ('EVENTS' in hdr):
        tracks = hdr['TRACKS']
        total_samples = hdr['TIME']['LENGTH']
        time_delta = hdr['TIME']['DELTA']
        machine_rep = hdr['MACHINE_REP']
        granularity = hdr['GRANULARITY']

        if 'FILE_VERSION' not in hdr:
            file_version = 0
        else:
            file_version = int(float(hdr['FILE_VERSION']))

        track_info, block_size = get_track_info(tracks, granularity,
                                                total_samples)[:2]
        event_track_info = _get_event_track_info(hdr['EVENTS']['TRACKS'])

        data_grabber = EventDataGrabber(data, track_info, event_track_info,
                                        block_size)
        packets, size, start, end = data_grabber.grab_block()
        if packets:
            if file_version < 2:
                raise OpalFileException('Event file version ' +
                                        str(file_version) + ' not supported')
            ofile = _open_file(base_name, 'wb')
            path = ofile.name

            while packets:
                next_vals = data_grabber.grab_block()
                next_packets = next_vals[0]
                _write_event_block(ofile, packets, event_track_info, start,
                                   end, size, machine_rep, file_version,
                                   (len(next_packets) == 0), time_delta)
                packets, size, start, end = next_vals

            ofile.flush()
            ofile.close()
    return path

def _get_event_track_info(tracks):
    """
    Returns a list of EventTrackInfo objects.

    Computes per track statistics for use when writing event data tracks.
    """

    track_info = []
    for track in tracks:
        fields = track['FIELDS']
        ti = EventTrackInfo()

        for field in fields:
            field_info = EventFieldInfo()
            field_info.name = field['NAME']
            field_info.m2k_format = field['FORMAT']

            # calculate the number of elements per sample
            field_info.elems_per_sample = 1
            if 'AXES' in track:
                axes = track['AXES']
                for i in xrange(1, len(axes)):
                    if 'LENGTH' not in axes[i]:
                        raise OpalFileException('LENGTH field of track ' +
                                                track + ' of axis ' + i +
                                                ' is missing')
                    field_info.elems_per_sample *= axes[i]['LENGTH']

            if field_info.m2k_format[0] == 'C':
                fmt = field_info.m2k_format[1:]
                # complex types are twice as large as normal types
                field_info.sample_size = (field_info.elems_per_sample *
                                          _m2k_size_dict[fmt] * 2)
                # we don't have native support in N.array for anything
                # but doubles and floats. we represent non native types as
                # a 2 dimension array so we have twice the number of elements
                if not _numpy_native_complex[fmt]:
                    field_info.elems_per_sample *= 2
                field_info.num_format = _type_to_cnumpy_dict[fmt]
            else:
                field_info.sample_size = (
                    field_info.elems_per_sample *
                    _m2k_size_dict[field_info.m2k_format])
                field_info.num_format = (
                    _type_to_numpy_dict[field_info.m2k_format])

            ti.fields.append(field_info)
            ti.packet_size += field_info.sample_size

        track_info.append(ti)
    return track_info


def _read_event_vals(block, curr, dformat, format_size, rep):
    """
    Returns a tuple.
    """

    temp = buffer(block, curr, format_size)
    vals = struct.unpack('%s%s' %
                          (_struct_endian[rep], dformat), temp)
    return vals

def _read_event_block(infile, offset, rep, version, tracks,
                      packets, length, event_track_info, options):
    """
    Returns a long.

    The return value is the offset in the file of the next event data
    block header. All read event data packets are added to the 'packets'
    dictionary.
    """
    if version < 3:
        mh_format = '4Q'
    else:
        mh_format = '2d2Q'
    size = struct.calcsize(mh_format)
    infile.seek(offset)
    string = infile.read(size)
    vals = struct.unpack('%s%s' % (_struct_endian[rep], mh_format), string)
    num_packets, next_offset = vals[2:]
    if next_offset == 0:
        block_size = length - offset
    else:
        block_size = next_offset - offset
    infile.seek(offset)
    block = infile.read(block_size)
    curr = size

    # read in all the track information first
    th_format = '2Q'
    th_size = struct.calcsize(th_format)
    for track_info in event_track_info:
        temp = buffer(block, curr, th_size)
        num_events, track_offset = (
            struct.unpack('%s%s' % (_struct_endian[rep],th_format), temp))
        track_info.num_events = num_events
        track_info.offset = track_offset - offset
        curr += th_size

    tracks_list = []
    for _ in tracks:
        tracks_list.append([])

    # next, read in the packet info
    packet_list = []
    ph2_format = '2Q'
    ph2_size = struct.calcsize(ph2_format)
    if version < 3:
        ph1_format = '2Q'
        ph1_size = struct.calcsize(ph1_format)
        for i in xrange(0, num_packets):
            vals1 = _read_event_vals(block, curr, ph1_format, ph1_size, rep)
            curr += ph1_size
            vals2 = _read_event_vals(block, curr, ph2_format, ph2_size, rep)
            curr += ph2_size
            start = _convert_to_real8(vals1[0])
            end = _convert_to_real8(vals1[1])
            packet = {'StartTime': start, 'EndTime': end,
                      'TRACKS': copy.deepcopy(tracks_list)}
            packet_list.append(packet)
            _append_packet(packets, vals2[1], packet)
    else:
        ph1_format = '2d'
        ph1_size = struct.calcsize(ph1_format)
        for i in xrange(0, num_packets):
            vals1 = _read_event_vals(block, curr, ph1_format, ph1_size, rep)
            curr += ph1_size
            packet_list.append({'StartTime': vals1[0], 'EndTime': vals1[1],
                                'TRACKS': copy.deepcopy(tracks_list)})
        for i in xrange(0, num_packets):
            vals2 = _read_event_vals(block, curr, ph2_format, ph2_size, rep)
            curr += ph2_size
            _append_packet(packets, vals2[1], packet_list[i])

    # now read in the fields
    for i in xrange(0, len(event_track_info)):
        track_info = event_track_info[i]
        fields = track_info.fields

        # first field is always TOA
        field_info = fields[0]
        toa_track_size = field_info.sample_size * track_info.num_events
        buf = buffer(block, curr, toa_track_size)
        curr += toa_track_size
        toa_arr = N.fromstring(buf, field_info.num_format)
        if rep != _native_rep:
            toa_arr = toa_arr.byteswap()
        assert len(toa_arr) == track_info.num_events

        # For each packet, walk through the TOA array. If the TOA falls
        # within this packet's start and end times, add the TOA field and
        # all fields at the same sample index to the packet.
        idx = 0
        for j in xrange(0, len(packet_list)):
            packet = packet_list[j]
            count = 0
            prev_idx = idx
            while idx < len(toa_arr):
                toa = toa_arr[idx]
                if (toa < packet['EndTime'] or
                    packet['EndTime'] <= packet['StartTime']):
                    count += 1
                    idx += 1
                else:
                    break
            if count:
                # append TOA fields first
                field_offset = track_info.offset
                size = count * fields[0].sample_size
                arr_offset = field_offset + prev_idx * fields[0].sample_size
                buf = buffer(block, arr_offset, size)
                field_arr = N.fromstring(buf, fields[0].num_format)
                if rep != _native_rep:
                    field_arr = field_arr.byteswap()
                packet['TRACKS'][i].append(field_arr)

                # offset of the first field after the TOA field
                field_offset += fields[0].sample_size * track_info.num_events
                for k in xrange(1, len(fields)):
                    field = fields[k]

                    # the total size we're going to read
                    size = count * field.sample_size

                    # offset into the field array
                    arr_offset = field_offset + prev_idx * field.sample_size

                    buf = buffer(block, arr_offset, size)
                    field_arr = _read_array_data(buf, field.m2k_format,
                                                 field.num_format, rep,
                                                 version, options)
                    tracks[i]['FIELDS'][k]['FORMAT'] = (
                        _convert_format(options,
                                        tracks[i]['FIELDS'][k]['FORMAT']))
                    packet['TRACKS'][i].append(field_arr)

                    # move to next field
                    field_offset += field.sample_size * track_info.num_events

            if idx == len(toa_arr):
                break

    return next_offset

def _convert_format(options, fmt):
    """
    Returns a string.

    Converts the given format to a new format based on the options given.
    If not conversion is required, the given format is returned.
    """
    if options and fmt[-1] == _long_long_m2k:
        if options & LONG_LONG_TO_INT:
            convert = 'L'
        elif options & LONG_LONG_TO_DOUBLE:
            convert = 'D'
        else:
            raise OpalFileException('Invalid options: ' + options)
        fmt = _long_long_convert[convert + fmt]
    return fmt

def _write_array_data(buf, data, ignored_m2k_format, rep, version):
    """
    Writes the numpy array, data, into buf.
    """
    if isinstance(data, basestring):
        buf.write(data)
    elif data.dtype == N.dtype(N.object_):
        assert isinstance(data[0], XMTime)
        new_data = N.zeros(len(data), N.uint64)
        for i in xrange(0, len(data)):
            new_data[i] = _convert_to_uint8(data[i].two_floats(), version > 2)
        if rep != _native_rep:
            new_data = new_data.byteswap()
        buf.write(new_data.tostring())
    else:
        if rep != _native_rep:
            data = data.byteswap()
        buf.write(data.tostring())

def _read_array_data(buf, dformat, num_format, rep, version, ignored_options):
    """
    Returns a numpy array for non-bit data. Returns a string for
    bit data.
    """
    if dformat == 'BIT':
        vals = str(buf)
    else:
        vals = N.fromstring(buf, num_format)
        if rep != _native_rep:
            vals = vals.byteswap()
        if dformat[0] == 'C':
            if not _numpy_native_complex[dformat[-1]]:
                assert len(vals) % 2 == 0
                vals = N.reshape(vals, (len(vals)/2, 2))
        elif dformat[0] == 'T':
            new_vals = N.zeros(len(vals), N.object_)
            for i in xrange(0, len(vals)):
                new_vals[i] = _convert_to_XMTime(vals[i], (version > 2))
            vals = new_vals
    return vals

def readevent(base_name, hdr, options=0):
    """
    Returns a list of packets.

    Reads the .evt file and returns all time packets in a single list.
    """
    packets = _read_event(base_name, hdr, options)
    keys = packets.keys()
    keys.sort()
    ret = []
    for key in keys:
        ret.extend(packets[key])
    return ret

def _read_event(base_name, hdr, options, packets = None):
    """
    Returns a dictionary.

    Reads the .evt file and returns all event data  packets in a dictionary.
    Each key in the dictionary is a sample index into the continuous data.
    The value for each key is a list of packets.
    """

    base_name = _add_extension(base_name, '.evt')
    if packets is None:
        packets = {}
    infile = _open_file(base_name, 'rb')
    if not infile:
        return packets

    length = os.stat(base_name).st_size

    events = hdr['EVENTS']
    if events:
        tracks = events['TRACKS']
        if tracks:
            machine_rep = hdr['MACHINE_REP']

            if 'FILE_VERSION' not in hdr:
                file_version = 0
            else:
                file_version = int(float(hdr['FILE_VERSION']))

            if file_version < 2:
                raise OpalFileException('Event file version ' +
                                        str(file_version) + ' not supported')

            if file_version < 2:
                comparison = length
            else:
                comparison = 0

            event_track_info = _get_event_track_info(tracks)

            next_offset = _read_event_block(infile, 0, machine_rep,
                                            file_version, tracks, packets,
                                            length, event_track_info, options)
            while (next_offset != comparison):
                next_offset = _read_event_block(infile, next_offset,
                                                machine_rep, file_version,
                                                tracks, packets, length,
                                                event_track_info, options)

    infile.close()
    return packets

def _insert_packets(hdr, data, packets, which_tracks=None):
    """
    Returns a list of lists.

    Inserts the packets into the continuous data. 'packets' is a
    dictionary where each key is a sample index into the continuous
    data. The value for each key is a list of packets.
    """

    if which_tracks is None:
        which_tracks = []

    if packets:
        sample_indexes = packets.keys()
        sample_indexes.sort()
        ret = []
        tracks = hdr['TRACKS']
        if data is None or tracks is None or len(tracks) == 0:
            # no data, just insert the packets
            for i in xrange(0, len(sample_indexes)):
                sample_index = sample_indexes[i]
                for packet in packets[sample_index]:
                    ret.append(packet)
        else:
            granularity = hdr['GRANULARITY']
            track_info, unused_block_size, track_names = (
                get_track_info(tracks, granularity))
            order = _get_track_indices(which_tracks, track_info, track_names)[1]
            prev_index = 0
            if sample_indexes[0] == 0:
                for packet in packets[0]:
                    ret.append(packet)
                range_start = 1
            else:
                range_start = 0
            for i in xrange(range_start, len(sample_indexes)):
                sample_index = sample_indexes[i]
                ret.append(_get_sub_arrays(data, track_info, order, prev_index,
                                           sample_index))
                for packet in packets[sample_index]:
                    ret.append(packet)
                prev_index = sample_index
            ret.append(_get_sub_arrays(data, track_info, order, prev_index, -1))
    else:
        if data is None:
            ret = []
        else:
            ret = data
    return ret

def _get_sub_arrays(data, track_info, order, start_idx, end_idx):
    """
    Returns a list of arrays.

    Gets a subset of each track in 'data', starting at sample 'start_idx' and
    ending at sample 'end_idx'.
    """
    lst = []
    for ii in xrange(0, len(order)):
        ti = track_info[order[ii]]
        start = start_idx * ti.elems_per_sample
        if end_idx == -1:
            end = len(data[ii])
        else:
            end = end_idx * ti.elems_per_sample

        if ti.m2k_format == 'BIT':
            if end % 8 or start % 8:
                raise OpalFileException('Cannot split BIT data on a ' +
                                        'non-byte boundary')
            end /= 8
            start /= 8

        arr = data[ii][start:end]
        lst.append(arr)
    return lst

def _get_track_indices(which_tracks, track_info, track_names):
    """
    Returns a set of indices and a list of ordered indices.

    Gets the set of indices and the ordered list of indices that correspond
    to 'which_tracks'.
    """
    order = []
    indices = set()
    if which_tracks:
        for wt in which_tracks:
            idx = -1
            if isinstance(wt, basestring):
                try:
                    idx = track_names[wt]
                except:
                    raise OpalFileException('Bad track name given: %s' % wt)
            else:
                try:
                    idx = int(wt)
                except:
                    raise OpalFileException('Bad track index: %s' % wt)
            if idx < 0 or idx >= len(track_info):
                raise OpalFileException('Bad track index: %s' % idx)

            if idx not in indices:
                order.append(idx)
                indices.add(idx)

        if not indices:
            raise OpalFileException('No tracks requested')
    else:
        for ii in range(len(track_info)):
            order.append(ii)
            indices.add(ii)

    return indices, order

def read(base_name, which_tracks=None, read_time=True, read_events=True,
         options=0):
    """
    Returns a dictionary.

    The header is imported as a dictionary.

    Continuous data is imported as a list, where each element of the list
    could be a time packet (a dictionary), an event data packet (a
    dictionary), or continuous data (a list of numpy arrays, or strings
    if it is bit data).

    All multi-element numeric data is returned as a numpy array.
    Complex double and float data (CD and CF) are supported directly by
    the numpy module; all other complex integer types are expressed by
    shaping the returned numpy array. Vectors of time values are
    returned as numpy arrays of XMTime objects.

    Reading of only specific tracks can be done by using the which_tracks
    argument. The tracks can be specified by name or by index. If which_tracks
    is an empty list or None, all tracks will be read.

    On platforms that don't support 8 byte integers, 8 byte integers are
    by default returned as Python long objects. The following options
    change that behavior:

    LONG_LONG_TO_INT: cast 8 byte integers to ints
    LONG_LONG_TO_DOUBLE: cast 8 byte integers to doubles

    Using the above options will also modify the header dictionary such
    that subsequent writing of the data will write the casted data.
    """

    if which_tracks is None:
        which_tracks = []

    hdr = readheader(base_name)
    data = readdata(base_name, hdr, which_tracks, options)

    time_packets = None
    event_packets = None

    if read_time:
        time_packets = readtime(base_name, hdr)
    if read_events:
        event_packets = _read_event(base_name, hdr, options)

    if event_packets:
        if len(data) > 0:
            data = _insert_packets(hdr, data[0], event_packets, which_tracks)
        else:
            data = _insert_packets(hdr, None, event_packets, which_tracks)
    if time_packets:
        time_packets.extend(data)
        data = time_packets

    # modify header to only include tracks requested
    if which_tracks:
        tracks = hdr['TRACKS']
        if tracks:
            total_samples = hdr['TIME']['LENGTH']
            granularity = hdr['GRANULARITY']
            track_info, unused_block_size, track_names = (
                get_track_info(tracks, granularity, total_samples))
            order = _get_track_indices(which_tracks, track_info,
                                       track_names)[1]
            hdr['TRACKS'] = [tracks[ii] for ii in order]

    return {'HEADER': hdr, 'DATA': data}

def write(base_name, table):
    """
    Returns a tuple.

    The tuple contains the names of the files that were written.
    """
    base_name = os.path.expanduser(os.path.expandvars(base_name))
    hdr_file = writeheader(base_name, table['HEADER'])
    data_file = writedata(base_name, table['HEADER'], table['DATA'])
    time_file = writetime(base_name, table['HEADER'], table['DATA'])
    event_file = writeevent(base_name, table['HEADER'], table['DATA'])

    ret = []
    if hdr_file is not None:
        ret.append(hdr_file)
    if data_file is not None:
        ret.append(data_file)
    if time_file is not None:
        ret.append(time_file)
    if event_file is not None:
        ret.append(event_file)

    return tuple(ret)

_default_hdr = {
    'FILE_VERSION': 3,
    'GRANULARITY': 1,
    'KEYWORDS': {},
    'MACHINE_REP': _native_rep,
    'NAME': '',
    'TIME': {
        'DELTA': 1.0,
        'KEYWORDS': {},
        'LENGTH': 0,
        'NAME': 'Time',
        'START': 0.0,
        'UNITS': 's'
        },
    'TIME_INTERPRETATION': {
        'AXIS_TYPE': 'CONTINUOUS'
        },
    'TRACKS': []
    }

def header(version=3, machine_rep=None, **kwargs):
    """
    Returns a header dict of the given version and byte order.
    'machine_rep' defaults to native byte order. This module only
    handles versions >= 2.

    Additional kwargs modify the top level Opal header dictionary keys
    from their default values, e.g. opalfile.header(GRANULARITY=1024).
    Caveats exist for the following keys:
      FILE_VERSION - always overridden by the 'version' argument.
      MACHINE_REP - overridden only if the 'machine_rep' argument is given.
      TRACKS - all TRACKS given and their associated track AXES are
        normalized by calling opalfile.add_track() and opalfile.add_axis().
    """
    hdr = copy.deepcopy(_default_hdr)
    hdr.update(**kwargs)
    hdr['FILE_VERSION'] = version
    if version < 2:
        raise OpalFileException("Version must be >= 2")
    if machine_rep:
        hdr['MACHINE_REP'] = machine_rep
    if 'TRACKS' in kwargs:
        # Normalize the list of track headers passed in. Their AXES
        # will be normalized by add_track()
        hdr['TRACKS'] = []
        for track in kwargs['TRACKS']:
            add_track(hdr, **track)
    return hdr

_default_track_hdr = {
    'NAME': 'Track 0',
    'AXES': [],
    'FORMAT': 'D',
    'KEYWORDS': {},
    'UNITS': '',
    }

def add_track(hdr, **kwargs):
    """
    Append a new TRACK header to the given OpalHeader dict TRACKS
    list. <kwargs> are used to update() the new TRACK header from its
    default, keys should be in all caps. If an AXES list is given, each
    of the new AXES headers are updated to contain all of the necessary
    keys. AXES[0] is always a link to the hdr['TIME'] axis, any
    value passed in is ignored and replaced.
    """
    track_hdr = copy.deepcopy(_default_track_hdr)
    track_hdr.update(**kwargs)
    if not 'NAME' in kwargs:
        track_hdr['NAME'] = 'Track %d' % len(hdr['TRACKS'])
    track_hdr['AXES'] = [hdr['TIME']]
    if 'AXES' in kwargs:
        for axis in kwargs['AXES'][1:]:
            add_axis(track_hdr, **axis)
    hdr['TRACKS'].append(track_hdr)
    return track_hdr

_default_axis_hdr = {
    'NAME': 'Axis 1',
    'START': 0.0,
    'DELTA': 1.0,
    'LENGTH': 1,
    'UNITS': '',
    'KEYWORDS': {},
    }

def add_axis(track_hdr, **kwargs):
    """
    Append an AXIS header to the given Opal TRACK header dict AXES
    list. <kwargs> are used to update() the new AXIS header from its
    defaults, keys should be in all caps.
    """
    axes = track_hdr['AXES']
    axis = copy.deepcopy(_default_axis_hdr)
    axis.update(**kwargs)
    if not 'NAME' in kwargs:
        axis['NAME'] = 'Axis %d' % len(axes)
    axes.append(axis)
    return axis

def get_nominal_time(data_object, sample, delta=None):
    """
    Returns a float.

    Returns the nominal time of the specified sample number. Will
    use any nominal time stamp information, if present; otherwise
    calculates the nominal time based on the time start in the
    header.
    """
    stamp = _get_nearest_nominal_stamp(data_object, sample)
    hdr = data_object['HEADER']
    if delta is None:
        delta = hdr['TIME']['DELTA']
    if stamp is not None:
        return stamp['Time'] + delta * (sample - stamp['FSO'])
    else:
        return get_simple_nominal_time(data_object, sample, delta)

def get_simple_nominal_time(data_object, sample, delta=None):
    """
    Returns a float.

    Returns the nominal time of the specified sample number without
    using any time packet packet information, i.e. only the header
    information is used.
    """
    hdr = data_object['HEADER']
    if delta is None:
        delta = hdr['TIME']['DELTA']
    start = hdr['TIME']['START']
    return start + delta * sample

def get_precision_time(data_object, track, sample, delta=None):
    """
    Returns a XMTime.

    Returns the precision time of the nth sample in track j. If no
    precision time information is available, None is returned.
    """
    stamp = _get_nearest_precision_stamp(data_object, track, sample)
    if stamp is None:
        return None
    if delta is None:
        hdr = data_object['HEADER']
        delta = hdr['TIME']['DELTA']
    return stamp['Time'] + delta * (sample - stamp['FSO'])

def _get_nearest_nominal_stamp(data_object, sample):
    """
    Returns a dictionary.

    Finds the nominal time stamp that is nearest to the sample and
    returns it. If no time stamp is found, None is returned.
    """
    found_stamp = 0
    stamp = None
    val = 0
    data = data_object['DATA']
    for item in data:
        if isinstance(item, dict) and ('Valid' in item) and item['Valid']:
            for curr_stamp in item['TimeStamps']:
                if not curr_stamp['RepresentsPTime']:
                    temp = abs(curr_stamp['FSO'] - sample)
                    if found_stamp:
                        if temp < val:
                            val = temp
                            stamp = curr_stamp
                    else:
                        val = temp
                        stamp = curr_stamp
                        found_stamp = 1
    return stamp

def _get_nearest_precision_stamp(data_object, track, sample):
    """
    Returns a dictionary.

    Finds the precision time stamp that is nearest to the sample and
    returns it. If no time stamp is found, None is returned.
    """
    found_stamp = 0
    stamp = None
    val = 0
    data = data_object['DATA']
    for item in data:
        if isinstance(item, dict) and ('Valid' in item) and item['Valid']:
            for curr_stamp in item['TimeStamps']:
                if (curr_stamp['RepresentsPTime'] and
                    curr_stamp['TrackNumber'] == track and
                    curr_stamp['VSO'] <= sample):
                    temp = long(abs(curr_stamp['FSO'] - sample))
                    if found_stamp:
                        if temp < val:
                            val = temp
                            stamp = curr_stamp
                    else:
                        val = temp
                        stamp = curr_stamp
                        found_stamp = 1
    return stamp
