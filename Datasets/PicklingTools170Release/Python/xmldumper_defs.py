

# Options for dictionaries -> XML
#  If XML attributes are being folded up, then you may
#  want to prepend a special character to distinguish attributes
#  from nested tags: an underscore is the usual default.  If
#  you don't want a prepend char, use XML_DUMP_NO_PREPEND option
XML_PREPEND_CHAR = '_'


# When dumping, by DEFAULT the keys that start with _ become
# attributes (this is called "unfolding").  You may want to keep
# those keys as tags.  Consider:
#
#   { 'top': { '_a':'1', '_b': 2 }} 
# 
# DEFAULT behavior, this becomes:
#   <top a="1" b="2"></top>       This moves the _names to attributes
#  
# But, you may want all _ keys to stay as tags: that's the purpose of this opt
#   <top> <_a>1</_a> <_b>2</b> </top>
XML_DUMP_PREPEND_KEYS_AS_TAGS = 0x100

# Any value that is simple (i.e., contains no nested
# content) will be placed in the attributes bin:
#  For examples:
#    { 'top': { 'x':'1', 'y': 2 }} ->  <top x="1" y="2"></top>
XML_DUMP_SIMPLE_TAGS_AS_ATTRIBUTES = 0x200

# By default, everything dumps as strings (without quotes), but those things
# that are strings lose their "stringedness", which means
# they can't be "evaled" on the way back in.  This option makes 
# Vals that are strings dump with quotes.
XML_DUMP_STRINGS_AS_STRINGS = 0x400

# Like XML_DUMP_STRINGS_AS_STRINGS, but this one ONLY
# dumps strings with quotes if it thinks Eval will return
# something else.  For example in { 's': '123' } : '123' is 
# a STRING, not a number.  When evalled with an XMLLoader
# with XML_LOAD_EVAL_CONTENT flag, that will become a number.
XML_DUMP_STRINGS_BEST_GUESS = 0x800

# Show nesting when you dump: like "prettyPrint": basically, it shows
# nesting
XML_DUMP_PRETTY = 0x1000

# Arrays of POD (plain old data: ints, real, complex, etc) can
# dump as huge lists:  By default they just dump with one tag
# and then a list of numbers.  If you set this option, they dump
# as a true XML list (<data>1.0/<data><data>2.0</data> ...)
# which is very expensive, but is easier to use with other
# tools (spreadsheets that support lists, etc.).
XML_DUMP_POD_LIST_AS_XML_LIST = 0x2000


# When dumping an empty tag, what do you want it to be?
# I.e., what is <empty></empty>  
# Normally (DEFAULT) this is an empty dictionary 'empty': {}
# If you want that to be empty content, as in an empty string,
# set this option: 'empty': ""
# NOTE: You don't need this option if you are using
# XML_DUMP_STRINGS_AS_STRINGS or XML_DUMP_STRINGS_BEST_GUESS
XML_DUMP_PREFER_EMPTY_STRINGS = 0x4000

# When dumping dictionaries in order, a dict BY DEFAULT prints
# out the keys in sorted/alphabetic order and BY DEFAULT an OrderedDict
# prints out in the OrderedDict order.  The "unnatural" order
# for a dict is to print out in "random" order (but probably slightly
# faster).  The "unnatural" order for an OrderedDict is sorted
# (because normally we use an OrderedDict because we WANTS its
# notion of order)
XML_DUMP_UNNATURAL_ORDER = 0x8000

# Even though illegal XML, allow element names starting with Digits:
# when it does see a starting digit, it turns it into an _digit
# so that it is still legal XML
XML_TAGS_ACCEPTS_DIGITS  = 0x80

# Allows digits as starting XML tags, even though illegal XML.
# This preserves the number as a tag.
XML_DIGITS_AS_TAGS = 0x80000


# When dumping XML, the default is to NOT have the XML header 
# <?xml version="1.0">:  Specifying this option will always make that
# the header always precedes all content
XML_STRICT_HDR = 0x10000

