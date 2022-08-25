
"""
Test to see if an instance of a Python object conforms to the specification
prototype.   This is similar to checking if an XML document conforms to an
XML DTD or schema.  For example:

  # EXAMPLE 1
  ##### At a simple level, we want to exactly match a prototype:
  >>> instance  = {'a':1, 'b':2.2, 'c':'three'}   # value to check
  >>> prototype = {'a':0, 'b':0.0, 'c':""}    # prototype to check against
  >>> if conforms(instance, prototype, True, EXACT_MATCH) :
  ...       should be true: all keys match, and value TYPES match

  (1) Note that the instance has all the same keys as the prototype, so it
      matches 
  (2) Note that on the prototype table, that the VALUES aren't important,
      it's only matching the the TYPE of the val

  # EXAMPLE 2
  #### We may not necesssarily need all keys in the prototype
  >>> instance1 = {'a':1, 'b':2.2, 'c':'three', 'd':777 }
  >>> prototype1= {'a':0, b:0.0 }
  >>> if conforms(instance1, prototype1, False, EXACT_MATCH) :
  ...    should be true: instance has all keys of prototype 

  (1) Note that the instance has more keys than the prototype, but that's
      okay because we specified exact_structure to be false.
  (2) by setting EXACT_MATCH, all the types of the values that are
      compared MUST match (not the value just the types of the values)

  # EXAMPLE 3
  #### If you just want the structure, but don't care about the
  #### types of the keys, use None in the prototype.
  >>> instance2 = {'a':1, 'b':2.2, 'c':'three'}
  >>> prototype2= {'a':None, 'b':None, 'c':None }
  >>> if conforms(instance2, prototype2, True, EXACT_MATCH) :
  ...    should be true, only comparing keys


  # EXAMPLE 4
  #### If you want to match value types, but want to be a little
  #### looser: sometimes your int is a long, sometimes an int_u4, etc.
  >>> instance3 = {'a':1, 'b':2L, 'c':'three'}
  >>> prototype3 ={'a':0, 'b':0, 'c':'three'}
  >>> if conforms(instance3, prototype3, true, LOOSE_MATCH) :
  ...    should be true because long(2) is a LOOSE_MATCH of int(2)


Conforms is the main routine described by this module.
Check and see if the given 'instance' of a Python dict, list, etc. conforms
to the given structure of a 'prototype' table.
To match, the instance must contain at least the same structure
as the prototype.  If the 'exact_structure' flag is set, the keys
and number of keys must match exactly.  If the 'match_types' flag is set, then
the type of the values of the keys are checked too.  
See below to see what kind of matching is done.
Finally, most of the time, you want to return True or False
to indicate the instance matches the spec---when debugging,
why an instance failed to match spec, you can have the
routine throw an exception with a MUCH more descriptive message
about why the conformance did not match.   Frequently, you won't
need that (and it's also expensive as it stringizes the 
tables of interest), so it's best you only use that feature
to help debug.

 Type match :

  # Specify how *values* (not keys) in a table or list must match
  EXACT_MATCH = 0       # int_4 must be int_4, etc

  LOOSE_MATCH = 1       # any int/long matches any int/long 
                        # any dict/OrderedDict matches any dict/OrderedDict
                        # any iterable matches any iterable

  LOOSE_STRING_MATCH=2  # As LOOSE_MATCH, but if the instance is 
                        # is a string but the prototype is of type x,
                        # call it a match.
                        # Consider: 
                        # instance = { 'a': "123" } prototype = {'a':1 }
                        # LOOSE_STRING_MATCH would match because 
                        # "123" is a string that *could* convert to an int.
"""

# Specify how *values* (not keys) in a table or list must match
EXACT_MATCH = 0      
LOOSE_MATCH = 1      
LOOSE_STRING_MATCH=2 

try :
    from collections import OrderedDict
except :
    OrderedDict = dict   # alias to allow us to make progress
    

# Main user routine
def conforms (instance, prototype, exact_structure=False,type_match=EXACT_MATCH,
              throw_exception_with_informative_message=False) :

    """ See if the instance conforms to the structure of the prototype. 
    Parameters:
    instance is the value to check for conformance
    prototype is the value that specifies the structure of a 'valid' entry
    exact_structure (if true) means the structure must match:
      TRUE: dict/OrderedDict must have the same number of *keys* w/ same names
            lists/tuples must have the same number of entries
      FALSE: 'subset check' for dict/OrderedDict to see if all *keys* of
              prototype are in instance  
             lists/tuples must have at least number of entries
    type_match specifies how to match the types of *values* (see enum above)
            dicts/OrderedDicts how to type match the *values* 
            lists/tuple specify how to match the values therein
    throw_exception_with_message: where there's an error ...
      FALSE: either return false (very cheap)
      TRUE:  throw an exception with an informative error message (expensive)
             (usually only set this to TRUE during debugging)
    """

    
    if None==prototype : return True   # No need for match
    if type(instance)==str and type_match==LOOSE_STRING_MATCH :
        return True
    
    elif isinstance(instance, dict) or isinstance(prototype, dict): 
        return table_match_(instance, prototype, exact_structure, type_match,
                            throw_exception_with_informative_message)

    elif isinstance(instance, str) or isinstance(prototype, str) :
        # str is iterable, but DO NOT want an iterable check
        return primitive_match_(instance, prototype, exact_structure,type_match,
                                throw_exception_with_informative_message)
    
    # See if it is iterable
    is_iterable = True
    try :
        it = iter(instance)
    except TypeError :
        is_iterable = False
    if is_iterable :
        return iterable_match_(instance, prototype, exact_structure, type_match,
                               throw_exception_with_informative_message)

    if is_primitive_type(instance) :
        return primitive_match_(instance, prototype, exact_structure,type_match,
                               throw_exception_with_informative_message)
    
    else :
        return userdefined_match_(instance,prototype,exact_structure,type_match,
                                  throw_exception_with_informative_message)


def userdefined_match_(instance,prototype,exact_structure,type_match,
                       throw_exception_with_informative_message) :
    # The only thing left is a user-defined class that's not iterable
    # So, we allow some type matching based on type.
    # TODO: should we iter through a dict?
    exact_match = (type(instance) == type(prototype))
    if exact_match : return True;
    
    if type_match==EXACT_MATCH :
        if throw_exception_with_informative_message :
            InformativeError_("Not exact match", instance, prototype,
                              type_match,
                              throw_exception_with_informative_message)
        else :
            return False
            
    # Assert: Not EXACT_MATCH, so some LOOSE match
    if type_match==LOOSE_STRING_MATCH and type(instance)==str :
        return True

    # Assert: all LOOSE_MATCH checking (LOOSE_STRING_MATCH falls to here)
    if isinstance(instance, type(prototype)) :
        return True;
    else :
        if throw_exception_with_informative_message :
            InformativeError_("Even a loose match can't match these two",
                              instance, prototype, type_match,
                              throw_exception_with_informative_message)
        else :
            return False

        
# Create an informative error message and throw an exception
# with that error message
def InformativeError_ (mesg, instance, prototype, 
                       exact_structure, type_match) :

    # stringize conform
    if type_match==EXACT_MATCH :
        conform = "EXACT_MATCH"
    elif type_match==LOOSE_MATCH :
        conform= "LOOSE_MATCH"
    elif type_match==LOOSE_STRING_MATCH :
        conform = "LOOSE_STRING_MATCH"
    else : 
        raise Exception("Illegal conform?"+str(int(type_match)))
  
    # message to output
    message = \
            "*********FAILURE TO MATCH instance against prototype:\n" \
            " instance="+str(instance)+" with type:"+str(type(instance))+"\n" \
            " prototype="+str(prototype)+" with type:"+str(type(prototype))+"\n" \
            " exact_structure="+str(exact_structure)+"\n" \
            " type_match="+conform+"\n"+ \
            mesg

    raise Exception(message)



def is_primitive_type (v) :
    """ Check to see if v is primitive.  Primitive in this context
    means NOT a container type (str is the exception):
    primitives type are: int, float, long, complex, bool, None, str 
    """
    return type(v) in {int:0, float:0, long:0, complex:0, bool:0, None:0, str:0}


def primitive_match_(v1, v2, exact_structure, type_match,
                     throw_exception_with_message ) :
    """See if two primitive types match: if v2 is None, then no need
    to check (None means 'don't match')."""
    # Doesn't matter: v2 (prototype) is None which means don't consider
    # the type 
    if None == v2 :
        return True

    # Make sure both primitive types
    if not is_primitive_type(v1) and not is_primitive_type(v2) :
        if throw_exception_with_message :
            InformativeError_("Neither Type is a primitive type", v1, v2, 
                              exact_structure, type_match)
        else :
            return False

    # For primitive types, no need for nested checking
    exact_match = (type(v1) == type(v2))
    if (exact_match) : return True  # easy case

    # Only if they don't match exactlty, figure out what to do
    if type_match == EXACT_MATCH :
        if exact_match==False and throw_exception_with_message :
            InformativeError_("Requested Exact Match of two primitive types "
                              "that didn't match", v1, v2,
                              exact_structure, type_match)
        else :
            return exact_match
    
    elif type_match == LOOSE_MATCH or type_match == LOOSE_STRING_MATCH :

        # long and int are close enough
        if type(v1)==int and type(v2)==long : return True
        if type(v2)==int and type(v1)==long : return True

        # Try convert other type
        if type_match == LOOSE_STRING_MATCH :
            # Sometimes, things are stringized and SHOULD be ints/floats
            # but they are still strings.  We let these through.
            if type(v1) == str  :
                return True
      
    
        # If we make it all the way here, then they didn't match anyway
        if throw_exception_with_message :
            InformativeError_("Even with loose match, they didn't match",
                              v1, v2, exact_structure, type_match)
        else :
            return False
    
    else :
        raise Exception("Unknown match type?")
  
    return False


# Helper function for just comparing Tables		       
def match_table_helper_ (instance, prototype, exact_structure, type_match,
                         throw_exception_with_message) :

    #print '****************match table helper', instance, prototype
    
    if exact_structure :
        # must have same number of elements, and each element must match
        if len(instance) != len(prototype) :
            if throw_exception_with_message :
                InformativeError_("instance and prototype do NOT have the same number of elements and we have requested an exact match",
                                  instance, prototype, exact_structure, type_match)
            else :
                return False
      
    # Assertion: just have to see if instance has required keys
    # specified in prototype
    for prototype_key, prototype_value in prototype.iteritems() :

        # If key is there, yay
        if prototype_key in instance :
            instance_value = instance[prototype_key]
            if None == prototype_value :  # no need to compare types
                # Just presence of key is good enough to keep moving
                pass
            else : # have to compare types
                one_value_result =  conforms(instance_value, prototype_value, 
                                             exact_structure, type_match, 
                                             throw_exception_with_message)
                # Whoops, this single key Conformance failed.
                if not one_value_result :
                    return False      # Conforms will give long error if needed
	

        # Key NOT there ... bad mojo! 
        else :
            if throw_exception_with_message :
                InformativeError_("The prototype has the key:"+str(prototype_key)+
                                  " but the instance does not.", instance, 
                                  prototype, exact_structure, type_match)
            else :
                return False
          
    return True


def table_match_ (v1, v2, exact_structure, type_match,
                  throw_exception_with_message) :
    """ See if two tables conform: front-end for match_table_helper_"""
    # No pattern match needed
    # print 'TABLE MATCH\n', v1, "\n", v2
    
    if None==v2 :
        return True
  
    # Only tables
    if not isinstance(v1, dict) or not isinstance(v2, dict) :
        if throw_exception_with_message :
            InformativeError_("table_match_ can only compare tables:",
                              v1, v2, exact_structure, type_match)
        else :
            return False
  
    # Same kind of table
    if type(v1)==type(v2) :
        return match_table_helper_(v1, v2, exact_structure, type_match, 
                                   throw_exception_with_message)

    if type_match == EXACT_MATCH :
        if throw_exception_with_message :
            InformativeError_("Not the same type for EXACT_MATCH", v1, v2,
                              exact_structure, type_match)
        else :
            return False  # because not same type!
    
    # Otherwise, handle slightly incompatible types        
    return match_table_helper_(v1, v2, exact_structure, type_match, 
                               throw_exception_with_message)
    

# helper for handling iterables
# Precondition: instance is iterable
def iterable_match_ (instance, prototype, exact_structure, type_match,
                    throw_exception_with_message) :

    # prototype is None, no match to be done
    if None==prototype : return True

    comp = (type(instance) == type(prototype))
    if type_match == EXACT_MATCH and comp == False :
        if throw_exception_with_message :
            InformativeError_("iterable types don't match exactly for an EXACT_MATCH",
                              instance, prototype, exact_structure, type_match)
        else :
            return False

    # Check if we can iterate through prototype:
    prototype_iterable = True
    try :
        it = iter(prototype)
    except TypeError :
        prototype_iterable = False
    if not prototype_iterable :
        if throw_exception_with_message :
            InformativeError_("prototype not iterable",
                              instance, prototype, exact_structure, type_match)
        else :
            return False

    # Exact structure implies same number of elements
    if exact_structure and len(instance) != len(prototype) :
        if throw_exception_with_message :
            InformativeError_("Iterables don't match: different lengths",
                              instance, prototype, exact_structure, type_match)
        else :
            return False
  
    # So, not exact structure.  But if the instance has fewer
    # elements than the prototype, they cannot match
    if len(instance) < len(prototype) :
        if throw_exception_with_message :
            InformativeError_("instance iterable has fewer elements than prototype iterable", 
                              instance, prototype, exact_structure, type_match)
        else :
            return False

    # Make sure all elements of prototype are found in instance
    instance_iter = iter(instance) 
    for prototype_value in prototype :

        # Advance instance as well
        instance_value = instance_iter.next()
        
        if None==prototype_value :
            continue # just checking existance, move on
        else :
            result = conforms(instance_value, prototype_value, exact_structure, 
                              type_match, throw_exception_with_message)
            if not result : # Check above handles better error message
                return False
  
    # made it here: all keys matched prototype well enough: call it good.
    return True





if __name__ == "__main__" :
    # EXAMPLE 1
    ##### At a simple level, we want to exactly match a prototype:
    instance  = {'a':1, 'b':2.2, 'c':'three'}   # value to check
    prototype = {'a':0, 'b':0.0, 'c':""}        # prototype to check against
    if conforms(instance, prototype, True, EXACT_MATCH) :
        print("should be true: all keys match, and value TYPES match")

    # EXAMPLE 2
    #### We may not necesssarily need all keys in the prototype
    instance1 = {'a':1, 'b':2.2, 'c':'three', 'd':777 }
    prototype1= {'a':0, 'b':0.0 }
    if conforms(instance1, prototype1, False, EXACT_MATCH) :
        print("should be true: instance has all keys of prototype")

    # EXAMPLE 3
    #### If you just want the structure, but don't care about the
    #### types of the keys, use None in the prototype.
    instance2 = {'a':1, 'b':2.2, 'c':'three'}
    prototype2= {'a':None, 'b':None, 'c':None }
    if conforms(instance2, prototype2, True, EXACT_MATCH) :
        print("should be true, only comparing keys")


    # EXAMPLE 4
    #### If you want to match value types, but want to be a little
    #### looser: sometimes your int is a long, sometimes an int_u4, etc.
    instance3 = {'a':1, 'b':2L, 'c':'three'}
    prototype3 ={'a':0, 'b':0, 'c':'three'}
    if conforms(instance3, prototype3, True, LOOSE_MATCH) :
        print("should be true because long(2) is a LOOSE_MATCH of int(2)")

