
# This requires the Python extension moduke "pyobjconvert" to be built:
# hea

"""
High-level tools to convert between XML and dicts using C extension modules
to gain additional speed.  See the documentation in xmlloader.py and
xmldumper.py for better descriptions of the interfaces below.

Note that the ``cxmltools`` does NOT have everything the ``xmltools`` has:
it only has the simplified wrappers (listed below). These are the same 
simplified wrappers that the xmldumper/xmlloader have as well.  
Really, the only thing you *don't* have are the classes (XMLDumper/XMLLoader)
that implements the conversions: all the functionality is still available, 
but through the easier-to-use wrappers.

For converting from XML to dict, the simplified wrappers are::

   ReadFromXMLFile(filename, options, array_disp, prepend_char);
   ReadFromXMLStream(stream, options, array_disp, prepend_char);
   ReadXMLString(xml_string, options, array_disp, prepend_char);

   defaults:
        options=XML_STRICT_HDR | XML_LOAD_DROP_TOP_LEVEL | XML_LOAD_EVAL_CONTENT
        array_disp=AS_NUMERIC
        prepend_char=XML_PREPEND_CHAR

For converting from dict to XML, the simplified wrappers are::

   WriteToXMLFile(dict_to_convert, filename, top_level_key, options,
                  array_disp, prepend_char)
   WriteToXMLStream(dict_to_convert, stream, top_level_key, options,
                    array_disp, prepend_char)
   WriteToXMLString(dict_to_convert, top_level_key, options,
                    array_disp, prepend_char)

   defaults:
        top_level_key=None   (this should probably always be "top" instead)
        options=XML_DUMP_PRETTY | XML_STRICT_HDR | XML_DUMP_STRINGS_BEST_GUESS
        array_disp=AS_NUMERIC
        prepend_char=XML_PREPEND_CHAR    

Note that the ``WriteToXMLString`` returns the string of interest, whereas the
WriteToXMLFile/Stream instead both write to the given entity (and return None).

"""

try :
    import pyobjconvert
except :
    m = """\n
The cxmltools module requires the Python C extension module pyobjconvert
to be built.  Check that pyobjconvertmodule.so is (1) built
and (2) on your PYTHONPATH.  See the README in PicklingTools141/PythonCExt
for information on building the C extension module and how to run it.
"""
    raise ImportError, m # propagate exception
    
from arraydisposition import *

from xmldumper_defs import *
WriteToXMLStream = pyobjconvert.CWriteToXMLStream
WriteToXMLFile    = pyobjconvert.CWriteToXMLFile
WriteToXMLString  = pyobjconvert.CWriteToXMLString

from xmlloader_defs import *
ReadFromXMLStream = pyobjconvert.CReadFromXMLStream
ReadFromXMLFile   = pyobjconvert.CReadFromXMLFile
ReadFromXMLString = pyobjconvert.CReadFromXMLString




