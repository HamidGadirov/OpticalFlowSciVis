#!/usr/bin/env python

""" Test to see how we handle all sorts of different XML when reading"""
from xmltools import *
import StringIO
import sys
import pretty
pretty.OTabRepr = 0  # To get OrderedTab printing from pretty instead of o{

try :
    from Numeric import array
except :
    from simplearray import SimpleArray as array

def singleTry (buffer, options, arr_disp) :
    v = None
    x = XMLLoader(buffer, options, arr_disp)
    string_exception_caught = False
    string_exception_text = ""
    
    try :
        sys.stdout.write("READING XML:\n"+str(buffer)+"\n")
        sys.stdout.flush()
        v = x.expectXML()
        
        print "Converted to Tab:" 
        pretty.pretty(v)
        
        print "BACK TO XML:"
        sys.stdout.flush()
        os = StringIO.StringIO()
        xd = XMLDumper(os, XML_DUMP_PRETTY, array_disposition=arr_disp)
        xd.XMLDumpValue(v)
        print os.getvalue()
        
    except Exception,e :
        print e
        sys.stdout.flush()
        string_exception_caught = True;
        string_exception_text = str(e)
        # print >> sys.stderr, "****************", str(e)
        # if "object of type" in str(e) : raise
        # raise   # DEBUG helper

    # Now test the streams: they should be EXACTLY the same
    # as the strings:  QUIETLY: if they are exactly the same,
    # no extra output should come out.  If they ARE different,
    # then we need to report that and immediately exit
    # so we can track down the problem
    sv = None
    stream_exception_caught = False
    stream_exception_text = ""

    ###os = buffer
    iss = StringIO.StringIO(buffer)
    xs = StreamXMLLoader(iss, options, arr_disp, XML_PREPEND_CHAR, True); # supress the warnings we normally generate so we don't see them twice in the output
    ###xs = XMLLoader(os, options, XML_PREPEND_CHAR, True); # supress the warnings we normally generate so we don't see them twice in the output
    try :
        sys.stdout.flush()
        sv = xs.expectXML()
    
        if (v != sv)  :
            # Stream version of output
            print "STREAM VERSION ERROR!!!!!!!!!!!!!!!!!!!!" 
            pretty.pretty(v)
            pretty.pretty(sv)
            sys.stdout.flush()
            print repr(v)==repr(sv)
            sys.exit(1)
            
    except Exception, e :
        # cout << e.what() << endl;
        stream_exception_caught = True
        stream_exception_text = str(e)
    
    # If exception
    if (string_exception_caught or stream_exception_caught) :
        if (stream_exception_caught==string_exception_caught) :
            if (stream_exception_text != string_exception_text) :
                print >> sys.stderr,  "ERROR: err messages for stream/string DO NOT match!"
                print >> sys.stderr,  string_exception_text
                print >> sys.stderr,  stream_exception_text
                sys.exit(1)
              
        elif (stream_exception_caught) :
            print >> sys.stderr,  "ERROR: stream version had exception, string did not!" 
            print >> sys.stderr,  string_exception_text
            print >> sys.stderr,  stream_exception_text
            sys.exit(1)
        elif (string_exception_caught) :
            print >> sys.stderr,  "ERROR: string version had exception, stream did not!"
            print >> sys.stderr,  string_exception_text
            print >> sys.stderr,  stream_exception_text
            sys.exit(1)
        else :
          # error message matched
          pass

    sys.stdout.flush()
    
def grindLoop (tests, options, arr_disp) :
    
    for t in tests :
        singleTry(t, options, arr_disp)
  

def simpleTests (options, arr_disp) :
  tests = [
    "<start>yo</start>",
    "<book><chapter>text</chapter></book>",
    "<book><chapter>text</chapter><date>200</date></book>",
    "<book><chapter>text</chapter><date>200</date><time><t>1</t><d>2</d></time></book>",
  ]
  grindLoop(tests, options, arr_disp)


def errorTests (options, arr_disp) :
  tests = [
    "<start>", # no end
    "<start></start>", # ok
    
    "<start><star>", # Whoops!  mistyped
    "<start></star>", # Whoops!  mistyped
    "<start>< /start>", # Whoops!  mistyped
    "< start></start>", # Whoops!  mistyped
    "<start </start>",
    "<start> </start/>",
    "<start> </start>",
    "<start>1</start>",
    "<start>   ' '    </start>",  #  TODO: Should this be error?
    "<start>   &apos; &apos;    </start>",
    "<start>  content  <nest>1</nest> more content </start>", # WARN
    "<start>  content  <nest>1</nest>  </start>", # WARN
    "<start>  <nest>1</nest>  content </start>", # WARN
    "<start>  <nest>1</nest>  </start>", # NO WARN ...just whitespace
    "<start> \n <nest>1</nest>   \n</start>", # NO WARN ...just whitespace

    "<start> \n <nest>1</nest>   \n</start>", # NO WARN ...just whitespace
  ]
  grindLoop(tests, options, arr_disp)


def simpleNestTests (options, arr_disp) :
  tests = [
    "<start>yo</start>",
    "<book><chapter>text</chapter></book>",
    "<book><chapter>text</chapter><date>200</date></book>",
    "<book><chapter>text</chapter><date>200</date><time><t>1</t><d>2</d></time></book>",
  ]
  grindLoop(tests, options, arr_disp)


def simpleListTests (options, arr_disp) :

  tests = [
    "<book><chapter>text1</chapter><chapter>text2</chapter></book>",
    "<book><chapter>text1</chapter><chapter>text2</chapter><chapter>text3</chapter></book>",
    "<book><chapter>text1</chapter><chapter> <p>text</p> </chapter><chapter>text3</chapter></book>",
    "<book><chapter>text1</chapter><chapter> <p>t1</p> <p>t2</p> <p></p> </chapter><chapter>text3</chapter></book>",
    "<book><chapter>text1</chapter><chapter> <p>t1</p> <p>t2</p> <pages>17</pages><date>2001</date> </chapter><chapter>text3</chapter></book>",
    "<book><chapter>text1</chapter><chapter> <p>t1</p> <p>t2</p> <pages>17</pages><date>2001</date> <p>ouro</p></chapter><chapter>text3</chapter></book>", # intersperced
    "<top><list__>0</list__><list__>1</list__><list__>2</list__></top><top><list__>2</list__><list__>3</list__><list__>4</list__></top>",
  ]
  grindLoop(tests, options, arr_disp)


def simpleAttrTests (options, arr_disp) :
  tests = [
    "<top a='1' b='2'>content</top>",
    "<top a='1' b='2'> <book></book></top>",
    "<top a='1' b='2'> <book date='2000'></book></top>",
    "<top a='1' b='2'> <book date='2000'></book><book key='2'></book><book>100</book></top>",
    "<top a='1' b='2'> <book date='2000'></book><book key='2'></book><book>100</book><book da='2'>666</book></top>",
  ]
  grindLoop(tests,options,arr_disp) # by default, make the attrs thing


def simpleXMLHeaderTests (options, arr_disp) :
  tests = [
    "<?xml version='1.0' encoding='UTF-8'?> <top>1</top>",
    "<?xml version='1.0' encoding='UTF-8'?> 1",
    "<?xml ?> 1",
    "<?xml version='1.0'><top>1</top>", # no end ?
    "<?xml version='1.0'?> <top>1</top>", # no end ?
    "<?xml version='1.0'?><top>1</top>", # no end ?
    "<xml version='1.0'?><top>1</top>", # no end ?
    "<?xml version='2.0'?><top>1</top>", 
    "   <?xml version='2.0'?><top>1</top>", 
    "   <?xml version='1.0' encoding='UTF-8'   ?><top>1</top>", 
    "   <?xml version='1.0' encoding='UTF-16'   ?><top>1</top>", 
    "   <?xml version='1.0' encoding='UTF-8'  ? ><top>1</top>", 
  ]
  grindLoop(tests, options, arr_disp) # by default, make the attrs thing



def simpleCommentTests (options, arr_disp) :
  tests = [
    "<?xml version='1.0'?>  <top>  <!--comment --> 1   </top>",
    "<?xml version='1.0'?>  <top>  <--comment --> 1   </top>", # bad
    "<?xml version='1.0'?>  <top>  <--comment -> 1   </top>",  # bad
    "<?xml version='1.0'?>  <top>  <!--comment -> 1   </top>", # bad
    "<?xml version='1.0'?>  <top>  1   <--comment!--></top>",
    "<?xml version='1.0'?>  <top>  <!--comment --> 1   <!--comment--></top>",
    "<?xml version='1.0'?>  <top>  <!--start chapter--><chapter>1</chapter><!--forgot! -->  <!--comment--></top>",
    "<?xml version='1.0'?>  <top>  <!--start chapter--><chapter>1</chapter><!--forgot! --> <chapter>2</chapter> <!--comment--></top>",
    "<?xml version='1.0'?>  <top>  <!--start chapter--><chapter>1</chapter><!--forgot! --> <chapter>2</chapter> <t>hello<!--why?--></t><!--because--><d><!--I can--></d><!--comment--></top>",
    "<?xml version='1.0'?>  <top>  <!---> 1   </top>", # bad
    "<?xml version='1.0'?>  <top>  <!----> 1   </top>", # okay
    "<?xml version='1.0'?>  <top><!--comment in empty--></top>", # okay
    "<?xml version='1.0'?>  <top>  <!--just spaces-->   </top>", # okay
    "<?xml version='1.0'?>  <top>     </top>", # okay
  ]
  grindLoop(tests, options, arr_disp)


def emptyContentTests (options, arr_disp) :
  tests = [ 
    "<top a='1' b='2'/>",
  ]
  grindLoop(tests, options, arr_disp)


def strictTests (options, arr_disp) :
  tests = [
    "<top a='1' b='2'/>",
    "<?xml version='1.0'?> <top>a</top>",
  ]
  grindLoop(tests, XML_STRICT_HDR, arr_disp)


def handleListsBetter (options, arr_disp) :
  tests = [
    "<top>\n"
    "  <int_4 arraytype__='l'>   -500, -376   , -253,   -129 ,-6,117,240,364, 487, 611    </int_4>\n"
    "  <real_4 arraytype__='d'>-500.0,-376,-253.0864219753086,-129.6296329629629,-6.172843950617278, 117.2839450617284, 240.7407340740741, 364.1975230864198, 487.6543120987654,     611.1111011111111 </real_4>\n"
    "  </top>",


    "<top>\n"
    "  <int_4 arraytype__='l'>-500,-376,-253,-129,-6,117,240,364,487,611</int_4>\n"
    "  <real_4 arraytype__='d'>-500.0,-376.5432109876543,-253.0864219753086,-129.6296329629629,-6.172843950617278,117.2839450617284,240.7407340740741,364.1975230864198,487.6543120987654,611.1111011111111</real_4>\n"
    "  </top>",


    "  <int_4 arraytype__='l'>-500,-376,-253,-129,-6,117,240,364,487,611</int_4>\n",

    "  <real_4 arraytype__='d'>-500.0,-376.5432109876543,-253.0864219753086,-129.6296329629629,-6.172843950617278,117.2839450617284,240.7407340740741,364.1975230864198,487.6543120987654,611.1111011111111</real_4>\n",

    "<top> <int_4 arraytype__='l'>1,2,3</int_4><int_4 arraytype__='d'>1,2,3</int_4></top>",

    
    "<top> <int_4>\n" 
    "        <list0__ type__='l'>1</list0__><list0__ type__='l'>2</list0__><list0__ type__='l'>3</list0__>\n  </int_4>\n"
    "      <int_4>\n"
    "        <list1__ type__='d'>1</list1__><list1__>2</list1__><list1__>3</list1__>\n   </int_4>\n"
    "</top>",

    "<top> <int_4>\n" 
    "        <list0__ type__='l'>1</list0__><list0__>2</list0__><list0__>3</list0__>\n  </int_4>\n"
    "      <int_4>\n"
    "        <list1__ type__='d'>1</list1__><list1__>2</list1__><list1__>3</list1__>\n   </int_4>\n"
    "</top>",

    "<top><l type__='l'>1</l><l type__='l'>2</l></top>",

    "<top><l type__='l'>1</l><l>2</l></top>",

    "<top><int_4 arraytype__='l'>1,2,3</int_4><int_4>17</int_4> </top>",

    "<top><i type__='l'>1</i><i type__='d'>2.2</i></top>", # needs warning
    "<top><i arraytype__='l'>1</i><i type__='l'>2.2</i></top>", 
    "<top><i>1</i><i type__='l'>2.2</i></top>", 
    "<top><i type__='l'>1</i><i arraytype__='l'>2.2</i></top>", 


  ]
  grindLoop(tests, options, arr_disp)


def handleDefaultList_ (options, arr_disp) :

  tests = [
    "<list0__ type='l'>1</list0__>",

    "<top><list__>0</list__><list__>1</list__><list__>2</list__></top>",

    "<top><i> <list0__ type__='l'>1</list0__></i> <i>stuff</i></top>",
    "<top><i> <list0__ type__='l'>1</list0__><list0__ type__='l'>2</list0__></i> <i>stuff</i></top>",

  ]
  grindLoop(tests, options, arr_disp); # by default, make the attrs thing



def handleraw (options, arr_disp) :

  # try ALL characters in an escape
  s = ""
  for ii in xrange(0, 256) :
      if chr(ii) in '&<>"\'' :
          s = s + "*"
      else :
          s = s + chr(ii)
      
  #s['&'] = '*';
  #s['<'] = '*';
  #s['>'] = '*';
  #s['"'] = '*';
  #s['\''] = '*';

  xml = "<top>" + s + "</top>" # ALL characters
  
  singleTry(xml, options, arr_disp)


def handleErrors (options, arr_disp) :

  tests = [
    "<top>\n<here>\n<",
    "<top>\n<here>\n\n\n\n\n\n",
    "<top>\n<here>\n</here>\n",
    "<top>\n<here>\n</here>",
  ]
  grindLoop(tests, options, arr_disp) # by default, make the attrs thing


def nestedLists (options, arr_disp) :
  tests = [
    "<top><empty type__='list'></empty> </top>",
  ]
  grindLoop(tests, options, arr_disp) # by default, make the attrs thing


def evalLists (options, arr_disp) :
  tests = [
    "<top> <c>1</c>  </top>",
  ]
  grindLoop(tests, options, arr_disp) # by default, make the attrs thing


def realTests (options, arr_disp) :
  tests = [
    "<top> <c>1234.5678</c>  </top>",
    "<top> <c>1234.56789123456789</c>  </top>",
    "<top> <c>1234.56789123456789Gunk</c>  </top>",
    "<top> <c>12345.123451234512345e10 garbage</c>  </top>",
    "<top> <c>12345.12345123451e10</c>  </top>",
    "<top> <c>123</c> </top>",
    "<top> <c>123 </c> </top>",
    "<top> <c>123L</c> </top>",
    "<top> <c>123 L</c> </top>",
    "<top> <c> 123</c> </top>",
    "<top> <c>9999999999999999999999999999999999999999999</c> </top>",
    "<top> <c>9999999999999999999999999999999999999999999L</c> </top>",
    "<top> <c>&quot;123&quot;</c> </top>", # keep quotes
    "<top> <c>&apos;123&apos;</c> </top>", # is just string
    "<top> <c> he said: &apos;123&apos; is here</c> </top>",
    "<top> <c> he said: &quot;123&quot; is here</c> </top>",
    "<!--comment here---> <top> <c> he said: &quot;123&quot; is here</c> </top>",
    "<!--comment here---> <!--- and another--><top> <c> he said: &quot;123&quot; is here</c> </top>",
    "<!--comment here---!> <!--- and another--!><top> <c> he said: &quot;123&quot; is here</c> </top>",
    "<top> <c> he said<!--What?-->: &quot;123&quot; is here</c> </top>",
    "<top> <c> he said<!--What?--><!--again?-->: &quot;123&quot; is here</c> </top>",
    "<top> <c> he said<!--What?--> <!--again?-->: &quot;123&quot; is here</c> </top>",

      "<! some crappy dtd .. CANNOT IGNORE unless there is an XML > <top> 1 </top>", # ERROR: need xml first
    "<?xml?> <! some crappy dtd .. we just ignore for now > <top> 1 </top>", 
    "<?xml?> <! some crappy dtd>  <!ignore me> <top> 1 </top>",


"<!DOCTYPE sgml [\n"
"  <!ELEMENT sgml (img)*>\n"
"   <!--\n"
"     the optional \"type\" attribute value can only be set to this notation.\n"
"   -->\n"
"  <!ATTLIST sgml\n"
"    type  NOTATION (\n"
"      type-vendor-specific ) #IMPLIED>\n"
"\n" 
"  <!ELEMENT img ANY> <!-- optional content can be only parsable SGML or XML data -->\n"
"   <!--\n"
"     the optional \"title\" attribute value must be parsable as text.\n"
"     the optional \"data\" attribute value will be set to an unparsed external entity.\n"
"     the optional \"type\" attribute value can only be one of the two notations.\n"
"   -->\n"
"  <!ATTLIST img\n"
"    title CDATA              #IMPLIED\n"
"    data  ENTITY             #IMPLIED\n"
"    type  NOTATION (\n"
"      type-image-svg |\n"
"      type-image-gif )       #IMPLIED>\n"
"\n"
"  <!--\n"
"    Notations are referencing external entities and may be set in the \"type\" attributes above,\n"
"    or must be referenced by any defined external entities that cannot be parsed.\n"
"  -->\n"
"  <!NOTATION type-image-svg       PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n"
"     \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n"
"  <!NOTATION type-image-gif       PUBLIC \"image/gif\">\n"
"  <!NOTATION type-vendor-specific PUBLIC \"application/VND.specific+sgml\">\n"
"\n"
"  <!ENTITY example1SVGTitle \"Title of example1.svg\"> <!-- parsed internal entity -->\n"
"  <!ENTITY example1SVG      SYSTEM \"example1.svg\"> <!-- parsed external entity -->\n"
"  <!ENTITY example1GIFTitle \"Title of example1.gif\"> <!-- parsed internal entity -->\n"
"  <!ENTITY example1GIF      SYSTEM \"example1.gif\" NDATA type-image-gif> <!-- unparsed external entity -->\n"
"]>\n"
"<sgml type=\"type-vendor-specific\">\n"
"  <!-- an SVG image is parsable as valid SGML or XML text -->\n"
"  <img title=\"example1SVGTitle;\" type=\"type-image-svg\">example1SVG;</img>\n"
"\n"
"  <!-- it can also be referenced as an unparsed external entity -->\n"
"  <img title=\"example1SVGTitle;\" data=\"example1SVG\" />\n"
"\n" 
"  <!-- a GIF image is not parsable and can only be referenced as an external entity -->\n"
"  <img title=\"example1GIFTitle;\" data=\"example1GIF\" />\n"
"</sgml>\n",
    
  ]
  grindLoop(tests, options, arr_disp)

def hexescape (options, arr_disp) :
    tests = [
        "<top><c>&#x20;</c></top>",
        "<top><c>&#x30;</c></top>",
        "<top><c>&#x20;</c></top>",
        "<top><c>&#x;</c></top>",
        "<top><c>&#xA1;</c></top>",
        "<top><c>&#xx;</c></top>",
        "<top><c>&#;;;</c></top>",
        "<top><c>&#;;;</c></top>",
        "<top><c>&#x20</c></top>",
        "<top><c>&#x20;&#x20;</c></top>",
        "<top><c>&#x30;&#x20;</c></top>",
        "<top><c>&#x20;&#x20;</c></top>",
        "<top><c>&#x;&#x20;</c></top>",
        "<top><c>&#xA1;&#x20;</c></top>",
        "<top><c>&#xx;&#x20;</c></top>",
        "<top><c>&#;;;&#x20;</c></top>",
        "<top><c>&#;;;&#x20;</c></top>",
        "<top><c>&#x20</c></top>",
        "<top><c>&#x20;&#x20;</c></top>",
        "<top><c>&#x20;&#x30;</c></top>",
        "<top><c>&#x20;&#x20;</c></top>",
        "<top><c>&#x20;&#x;</c></top>",
        "<top><c>&#x20;&#xA1;</c></top>",
        "<top><c>&#x20;&#xx;</c></top>",
        "<top><c>&#x20;&#;;;</c></top>",
        "<top><c>&#x20;&#;;;</c></top>",
        "<top><c>&#x20;&#x20</c></top>",
    ]
    grindLoop(tests, options, arr_disp);
  

def fullTests (options, arr_disp=ARRAYDISPOSITION_AS_NUMERIC) :
  print "**************** OPTIONS: all defaults simple test"
  simpleTests(options, arr_disp)

  print "**************** OPTIONS: all defaults error test"
  errorTests(options, arr_disp)

  print "**************** OPTIONS: all defaults nest test"
  simpleNestTests(options, arr_disp)

  print "**************** OPTIONS: all defaults list test"
  simpleListTests(options, arr_disp)

  print "**************** OPTIONS: all defaults"
  simpleAttrTests(options, arr_disp)

  print "**************** OPTIONS: XML_LOAD_DROP_ALL_ATTRS"
  simpleAttrTests(options | XML_LOAD_DROP_ALL_ATTRS, arr_disp)
  
  print "**************** OPTIONS: XML_LOAD_UNFOLD_ATTRS" 
  simpleAttrTests(options | XML_LOAD_UNFOLD_ATTRS, arr_disp)

  print "**************** OPTIONS: XML_LOAD_TRY_TO_KEEP_ATTRIBUTES_WHEN_NOT_TABLES"
  simpleAttrTests(options | XML_LOAD_TRY_TO_KEEP_ATTRIBUTES_WHEN_NOT_TABLES, arr_disp)

  print "**************** XML header tests"
  simpleXMLHeaderTests(options, arr_disp)

  print "**************** comments test: "
  simpleCommentTests(options, arr_disp)

  print "**************** empty Content test: "
  emptyContentTests(options, arr_disp)

  print "**************** empty strict test: "
  strictTests(options, arr_disp)
  
  print "**************** handle lists better"
  handleListsBetter(options, arr_disp)

  print "**************** handle default lists (list__)"
  handleDefaultList_(options, arr_disp)

  print "**************** handle raw"
  handleraw (options, arr_disp)

  print "**************** longer errors"
  handleErrors (options, arr_disp)

  print "**************** nested lists"
  nestedLists (options, arr_disp)

  print "***************** Eval lists"
  evalLists(XML_LOAD_EVAL_CONTENT, arr_disp) 

  print "***************** Several reals"
  realTests(XML_LOAD_EVAL_CONTENT, arr_disp)

  print "***************** Hex escape"
  hexescape(options, arr_disp)

if __name__ == "__main__" :

  arr_disp = ARRAYDISPOSITION_AS_NUMERIC_WRAPPER
  try :
      import Numeric
      arr_disp = ARRAYDISPOSITION_AS_NUMERIC
  except :
      pass

  try :
      import numpy
      arr_disp = ARRAYDISPOSITION_AS_NUMPY
  except :
      pass
  
  #fullTests(0)   # With Tabs
  #fullTests(XML_LOAD_USE_OTABS)  # With OTabs
  #fullTests(XML_LOAD_EVAL_CONTENT)


  if OrderedDict==dict :
      print 'Running this test on a python that does not support OrderedDicts'
      print '... the diff will not pass, but the only difference should be'
      print '... { } vs. OrderedDict([])'
  
  fullTests(0, arr_disp)   # With Tabs
  fullTests(XML_LOAD_USE_OTABS, arr_disp)  # With OTabs
  fullTests(XML_LOAD_EVAL_CONTENT, arr_disp)

  fullTests(0, ARRAYDISPOSITION_AS_LIST)   # With Tabs
  fullTests(XML_LOAD_USE_OTABS, ARRAYDISPOSITION_AS_LIST)  # With OTabs
  fullTests(XML_LOAD_EVAL_CONTENT, ARRAYDISPOSITION_AS_LIST)   

