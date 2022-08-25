

###################### OPTIONS for XML -> dictionaries

#  ATTRS (attributes on XML nodes) by default becomes
#  separate dictionaries in the table with a 
#  "__attrs__" key.  If you choose to unfold, the attributes
#  become keys at the same level, with an underscore.
#  (thus "unfolding" the attributes to an outer level).
#  
#  For example:
#    <book attr1="1" attr2="2>contents</book>
#  WITHOUT unfolding  (This is the DEFAULT)
#    { 'book' : "contents",
#      '__attrs__' : {'attr1'="1", "attr2"="2"}
#    }
#  WITH unfolding:  (Turning XML_LOAD_UNFOLD_ATTRS on)
#    { 'book' : "contents",
#      '_attr1':"1", 
#      '_attr2':"2", 
#    }
XML_LOAD_UNFOLD_ATTRS = 0x01


#  When unfolding, choose to either use the XML_PREPEND character '_'
#  or no prepend at all.  This only applies if XML_LOAD_UNFOLD_ATTRS is on.
#    <book attr1="1" attr2="2>contents</book>
#  becomes 
#   { 'book': "content", 
#     'attr1':'1',
#     'attr2':'2'
#   }
#  Of course, the problem is you can't differentiate TAGS and ATTRIBUTES 
#  with this option 
XML_LOAD_NO_PREPEND_CHAR = 0x02

#  If XML attributes are being folded up, then you may
#  want to prepend a special character to distinguish attributes
#  from nested tags: an underscore is the usual default.  If
#  you don't want a prepend char, use XML_LOAD_NO_PREPEND_CHAR option
XML_PREPEND_CHAR = '_'


#  Or, you may choose to simply drop all attributes:
#  <book a="1">text<book>
#    becomes
#  { 'book':'1' } #  Drop ALL attributes
XML_LOAD_DROP_ALL_ATTRS = 0x04

#  By default, we use Dictionaries (as we trying to model
#  key-value dictionaries).  Can also use ordered dictionaries
#  if you really truly care about the order of the keys from 
#  the XML
XML_LOAD_USE_OTABS = 0x08

#  Sometimes, for key-value translation, somethings don't make sense.
#  Normally:
#    <top a="1" b="2">content</top>
#  .. this will issue a warning that attributes a and b will be dropped
#  becuase this doesn't translate "well" into a key-value substructure.
#    { 'top':'content' }
# 
#  If you really want the attributes, you can try to keep the content by setting
#  the value below (and this will supress the warning)
#  
#   { 'top': { '__attrs__':{'a':1, 'b':2}, '__content__':'content' } }
#  
#  It's probably better to rethink your key-value structure, but this
#  will allow you to move forward and not lose the attributes
XML_LOAD_TRY_TO_KEEP_ATTRIBUTES_WHEN_NOT_TABLES = 0x10

#  Drop the top-level key: the XML spec requires a "containing"
#  top-level key.  For example: <top><l>1</l><l>2</l></top>
#  becomes { 'top':[1,2] }  (and you need the top-level key to get a 
#  list) when all you really want is the list:  [1,2].  This simply
#  drops the "envelope" that contains the real data.
XML_LOAD_DROP_TOP_LEVEL = 0x20

#  Converting from XML to Tables results in almost everything 
#  being strings:  this option allows us to "try" to guess
#  what the real type is by doing an Eval on each member:
#  Consider: <top> <a>1</a> <b>1.1</b> <c>'string' </top>
#  WITHOUT this option (the default) -> {'top': { 'a':'1','b':'1.1','c':'str'}}
#  WITH this option                  -> {'top': { 'a':1, 'b':1.1, 'c':'str' } }
#  If the content cannot be evaluated, then content simply says 'as-is'.
#  Consider combining this with the XML_DUMP_STRINGS_BEST_GUESS
#  if you go back and forth between Tables and XML a lot.
#  NOTE:  If you are using Python 2.6 and higher, this uses ast.literal_eval,
#         which is much SAFER than eval.  Pre-2.6 has no choice but to use
#         eval.
XML_LOAD_EVAL_CONTENT = 0x40

# Even though illegal XML, allow element names starting with Digits:
# when it does see a starting digit, it turns it into an _digit
# so that it is still legal XML
XML_TAGS_ACCEPTS_DIGITS = 0x80

# Allows digits as starting XML tags, even though illegal XML.
# This preserves the number as a tag.
XML_DIGITS_AS_TAGS = 0x80000

#  When loading XML, do we require the strict XML header?
#  I.e., <?xml version="1.0"?>
#  By default, we do not.  If we set this option, we get an error
#  thrown if we see XML without a header
XML_STRICT_HDR = 0x10000



