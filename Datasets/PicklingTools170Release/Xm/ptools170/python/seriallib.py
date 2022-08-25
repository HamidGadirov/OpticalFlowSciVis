
# TODO: (sat 1-25-2011) Handle hierarchies via dotted or slashed
# paths in INI files (nested dicts in dict format).

# TODO: (sat 1-25-2011) Add functions for going to and from
# pretty-printed dicts. Note the pretty() function in PTOOLS.

# TODO: (sat 1-Feb-2011) Need to design name mappings for going to/from
# filesystem

MAGIC_KEY = '__SERIALLIB_SERIALIZE__'

def dict_to_ini(d):
    """
    Given a dictionary, return a ConfigParser instance reflecting the dict's
    structure.
    
    Currently only handles dicts that look like .ini files; that is:
    
    { 'section name 1': { k1: v1, ..., kN: vN}, 
      'section name 2': ...
      ...
    }
    
    N.B.: The returned ConfigParser will store the values as they were given,
    NOT convert them to strings. This means that you will need to do 'raw'
    gets to bypass the value interpolation logic that assumes all values are
    strings.  getint() and getfloat() will also fail on non-string
    values. (This caution applies to Python 2.4; it is unknown whether it
    still applies in newer Pythons.)
    """
    from ConfigParser import ConfigParser
    cp = ConfigParser()
    for section, options in d.items():
        cp.add_section(section)
        # This will fail if the structure of the dictionary does not map onto
        # an INI format. We could potentially support "section-less" dicts by
        # introducing a mapping to something like a 'default' section. It
        # would have to work round-trip.
        for (opt, val) in options.items():
            # Note that we are setting a potentially non-string value
            # here. This will require us to do a "raw" get for round-trip
            # conversions.  Otherwise the interpolator logic will be called
            # in ConfigParser, which assumes values are strings.  instance.
            cp.set(section, opt, val)
    return cp


def ini_to_dict(ini_file):
    """
    Given a ConfigParser instance or pathname to an INI-formatted file,
    return a dict mapping of the ConfigParser. Each section becomes a
    top-level key.
    """
    from ConfigParser import ConfigParser
    if isinstance(ini_file, str):
        cp = ConfigParser()
        # If we need to read multiple files to fill out the ConfigParser, let
        # the caller do it. This is a convenience shortcut for a common case.
        cp.read(ini_file)
    else:
        # Note no further checking for type. We'll let the sections() inspector
        # raise an exception if ini_file is not a ConfigParser
        cp = ini_file
    
    d = {}
    for section in cp.sections():
        # Do a raw get to avoid interpolation because values may not be strings
        # if this ConfigParser instance was not read from a file.
        #
        # This has the unfortunate consequence of NOT working on
        # RawConfigParser instances, but our general usage pattern only
        # employs ConfigParser, not its Raw or Safe variants.
        d[section] = dict(cp.items(section, raw=True))
    return d


def dict_to_kvstring(d, sep=':'):
    """
    Given a flat dictionary, return a simple key-value string. Each entry in the
    dict gets one line in the output.
    
    Examples:
    
      { 'key1': 'the first value',
        2: 'the second value' }
        
        ->
      'key1:the first value\n2:the second value\n'
    
    If optional 'sep' is given, it is used to separate keys and values. In any
    case, 'sep' should probably not be in the string representation of any key!
    
    If the input dict is not flat, the behavior is undefined.
    """
    strs = []
    items = d.items()
    # Sort for consistency and ease of verification
    items.sort(key=lambda x:x[0])
    for k, v in items:
        strs.append('%s%s%s' % (k, sep, v))
    return '\n'.join(strs)


def filter_blanks_and_comments(s, sep='#') :
    """
    Helper function for kvstring_to_dict.  Given 's',
    one string of the form 'key:value\nkey2:value2\n'
    (with any number of key value strings), turn it into
    a list of lines, where the blanks on the ends have
    been filtered, and any comments at the end-of-the-line
    are filtered.
       'tarlec:  herbert  # or is just herb?\n nesman: lester # or les?\n'
        becomes
       ['tarlec:  herbert','nesman: lester']
    """
    result = []
    lines = s.splitlines()  
    for line in lines:
        line = line.split(sep, 1)[0].strip()
        if not line :
            continue
        result.append(line)
    return result

def kvstring_to_dict(s, sep=':'):
    """
    Given a multi-line string containing key-value pairs, return a flat dict
    representing the same relationships. Whitespace around the separator is
    ignored.
    
    Expects one entry per line in the input.
    
    Note that all keys and values are interpreted as strings upon return.
    """
    # IDEA (sat 15-Jun-2011): Add a special "guess" flag. Use regexp like
    #
    #   \w+\s*(\W+)\s*\w+
    #
    # The called-out group is the separator candidate. Histogram all
    # the candidates. If there is only one nonempty bin, that's the
    # separator. Otherwise, could warn and choose the most likely, or
    # just fail.
    #
    # If we don't want to require a single common separator, we could
    # even allow it to change from line to line, using the "not word"
    # criterion. With a little care, this would even allow whitespace
    # as the separator.

    # IMPROVEMENT (sat 15-Jun-2011): If sep is a regexp object, do the right
    # thing.
    
    # NOTE: (sat 6/13/2011) If we want to support multiline values, we'll have
    # to do some design work both here and in dict_to_kvstring.
    # from satlib import filter_blanks_and_comments
    lines = filter_blanks_and_comments(s)
    items = []
    for line in lines:
        k, v = line.split(sep, 1)
        # This actually removes more than just whitespace around the separator,
        # but it seems a wise precaution for normalizing input.
        k = k.strip()
        v = v.strip()
        items.append((k,v))
    return dict(items)
    
# IMPROVEMENT: (sat 2/2/2011) There is a use case for converting dicts
# to objects regardless of the object's class type, in effect
# automatically using objects as syntactic sugar for dicts. This would
# require a 'force' or 'all' kind of flag that prefers the dict to the
# output of get_serialized_attributes.
def dict_to_instance(d, obj):
    """
    Apply the given dictionary to the given object by assigning attributes in
    the object for each top-level key in the dict.
    
    Only those keys in 'd' which match names in get_serialized_attributes(obj)
    will be applied. Extras will be ignored. 
    """
    # TODO: (sat 1-25-2011) Need to handle nested specs if the target does not
    # contain the top-level nested object.
    descs = get_serialized_attributes(obj)

    for ad in descs:
        if not d.has_key(ad.name):
            continue
        val = d[ad.name]
        ad.deserialize(obj, val)        
            
        
def instance_to_dict(obj):
    """
    Snapshot the given object as a dictionary. The resulting dictionary can be
    used as a memento to for the important state.
    """
    descs = get_serialized_attributes(obj)
    # (sat 2/1/2011): This logic is awkward because of the way things are
    # factored. The basic design decision here is for AttributeDescriptor to
    # be general-purpose and complete, but not to have a nice syntax. This
    # function and its counterpart are the intended nice syntax.
    ret = {}
    for d in descs:
        try:
            ret[d.name] = d.serialize(obj)
        except AttributeError:
            # This will happen when there's an attribute which should be
            # serialized in general, but which is not present in this instance.
            pass
    return ret


def dict_to_filesystem(d, root_path):
    """
    Given a dictionary, serialize it into a filesystem path. Each key will
    create a file in the filesystem, whose contents will be the value of that
    key.
    """
    # TODO: (sat 1/25/2011) Need to handle some special cases here,
    # particularly BLUE files and packed structs


def filesystem_to_dict(root_path):
    """
    
    """
    # TODO: (sat 1/25/2011) Handle symlinks, BLUE files, packed structs


class AttributeDescriptor(object):
    # If it's helpful in practice, we could add a **kwargs here so that
    # instances can be constructed with a minimal invocation. This will
    # complicate the logic somewhat as we will have to default arg to a guard
    # value, and throw an exception if it's still the default but no kwargs
    # were specified.
    def __init__(self, arg):
        """
        Describe an attribute to be serialized on classes and instances. 'arg'
        may be any of:
        
        string 'x': 
            save and apply attribute 'x' without type coercion
        
        dict { 'name': <str>,
               'deserialize_func': <callable>,
               'serialize_func':   <callable>,
               'default_value':    <val>,
               'setter_func_name': <str>,
               'getter_func_name': <str>,
             }:
            Save and apply the named attribute. 
            Other items in the dict, besides 'name', are optional.

            If deserialize_func is present and not None, it is a callable
              that is given the raw value from the intermediate
              representation. Examples are 'float' and lambda s:
              struct.unpack('f', s)

            If serialize_func is present and not None, it is a callable that
              is given the attribute value from the instance and which should
              return a string- izable representation of the value. If this is
              None, then str() will be called upon serialization.

            If default_value is present, it is used when the name is not
              present during deserialization.
              
            If present, setter_func_name and getter_func_name are the NAMES of
              functions which should be called in the target object during
              deserialization and serialization, respectively. Although they
              may be specified in any combination with serialize_func and
              deserialize_func, note that you can always implement
              serialization in the getter function alone, and deserialization
              in the setter function.
              
        tuple ('x', deserialize_func, serialize_func, default_value):
            Save and apply attribute named 'x'.
            Other items in the tuple are optional, and have the same meaning
            as in the dict representation. Although items are optional, they
            follow a strict positional mapping. If you want to specify
            default_value without specifying serialize_func, either pass in
            None for serialize_func, or use an alternate representation such
            as a dict.
        
        Another AttributeDescriptor instance: copy the settings from 'arg'
        """
        self._raw_arg = arg
        self.name = None
        self.deserialize_func = None
        # The default behavior of str() is applied in serialize(), q.v.
        self.serialize_func = None
        self.default_value = None
        self.has_default = False
        
        # These functions are stored by name rather than as function objects
        # for two reasons. First, we wish to allow the latest possible
        # binding of behaviors in general, so that a function named set_x can
        # be called polymorphically on different class types. Second, it can
        # become tricky to distinguish between functions, bound methods, and
        # unbound methods, and to apply all of them correctly to an
        # instance. With the name approach, we always get a bound method and
        # this problem is avoided.
        self.setter_func_name = None
        self.getter_func_name = None
        
        if isinstance(arg, str):
            self.name = arg
            return
        
        if isinstance(arg, AttributeDescriptor):
            # We completely control this class, so it's fine to jump down into
            # the __dict__ implementation of an object's data members.
            self.__dict__.update(arg.__dict__)
            return
        # These next branches would really benefit from the ideas in the
        # Abstract Base Class PEP (PEP 3119). But specific types are good
        # enough for the initial use cases.
        if isinstance(arg, dict):
            if not arg.has_key('name'):
                raise ValueError('AttributeDescriptor constructor requires field "name": %r'
                                  % arg)
            # Note that this will silently pass through extra dictionary
            # entries, storing them in this object's namespace. Not what we
            # want, but since this is undefined behavior, we don't need to fix
            # it just yet.
            self.__dict__.update(arg)
            if arg.has_key('default_value'):
                # Want to allow default to span all values, so need another
                # flag to record whether to apply it
                self.has_default = True
            return
        
        # Now the only allowed types left are iterables like tuples and
        # lists.  Turn both into lists so that we can use pop(). Don't worry
        # about other objects being converted to list incorrectly, because
        # that's undefined behavior anyway.
        arg = list(arg)
        
        # Now this is subtle, but important. Observation shows that calling
        # list(arg), even when arg is already a list, creates a new object. So
        # we can safely modify larg without affecting arg. We depend on this
        # below. Since it's subtle, important, and not derived from
        # documentation, assert to the rescue!
        assert(id(arg) != id(self._raw_arg))

        positions = ['name', 'deserialize_func', 'serialize_func', 'default_value']
        while arg and positions:
            attr = positions.pop(0)
            val  = arg.pop(0)
            setattr(self, attr, val)
        # Here's where we rely on the subtlety mentioned above, that list(arg)
        # gives a different object than arg even if arg is already a list. So
        # self._raw_arg was unchanged by the pops (its length in particular). 
        if len(self._raw_arg) >= 4:
            self.has_default = True
        
    
    def __str__(self):
        # Since self._raw_arg might be a tuple, we need to wrap it in a tuple
        # to avoid interpolating it into the string
        return str('<AttributeDescriptor(%r)>' % (self._raw_arg,))
    

    def serialize(self, obj):
        """
        Retrieve the attribute described by 'self' from obj, serializing it as
        appropriate.
        
        If self has a getter_func_name, the callable of that name is looked up
        in 'obj' and its return value retrieved. Default handling is bypassed.
        
        Otherwise, if obj has no attribute named self.name, and self has a
        default value, the default value is retrieved.

        If self has a serialize_func, it is called with the value retrieved from
        the object, and its return value becomes the return value of this
        function.
        """
        if self.getter_func_name:
            getter = getattr(obj, self.getter_func_name)
            val = getter()
        else:
            # This is structured the way it is so that AttributeError is
            # raised for us by the getattr call when appropriate
            if not hasattr(obj, self.name) and self.has_default:
                val = self.default_value
            else:
                val = getattr(obj, self.name)
        
        # (sat 2/1/2011): We have the choice of storing serialize_func
        # as str if not given in the constructor, or applying a
        # default behavior here. I chose the latter because it keeps a
        # distinction between default behavior and the user explicitly
        # specifying str as a serialize_func, and because it allows
        # the latest possible binding.
        serialize_func = self.serialize_func
        if not serialize_func:
            serialize_func = str
            
        val = serialize_func(val)
            
        return val
    
    
    def deserialize(self, obj, val):
        """
        Apply the given value 'val' to instance 'obj', deserializing it as
        appropriate.
        
        If self has a deserialize_func, it is called with 'val' and its return
        value is assigned instead of 'val'.

        If self has a setter_func_name,  the callable of that name is looked up
        in 'obj' and called with the (possibly deserialized) 'val'.
        """
        if self.deserialize_func:
            val = self.deserialize_func(val)
        if self.setter_func_name:
            setter = getattr(obj, self.setter_func_name)
            setter(val)
        else:
            setattr(obj, self.name, val)
    
    
    def as_dict(self):
        """
        Return a dict describing self, suitable for use as an
        AttributeDescriptor constructor argment.
        """
        # name is required
        ret = {'name': self.name}
        # functions are optional; value None means no function 
        for k in ['serialize_func', 'deserialize_func']:
            func = getattr(self, k)
            if func:
                ret[k] = func
        # default_value can take on any value, including None, but in
        # dictionary form the presence of the key determines whether we have
        # a default
        if self.has_default:
            ret['default_value'] = self.default_value
        return ret

    
        
def set_serialized_attributes(klass, *attrs):
    """
    Annotate a class or instance with a set of attributes which should be
    serialized and deserialized in apply_dict_to_instance and
    dict_from_instance.
    
    Following arguments may be AttributeDescriptor instances or anything that
    can be used to construct an AttributeDescriptor.
    """
    # (sat 1/26/2011) We make a policy decision for what to do if any
    # attribute is not a member of the class. We often populate
    # instances dynamically, so an error is probably not called for at
    # this time; rather we wait until we are actually serializing an
    # instance.
    to_save = [AttributeDescriptor(a) for a in attrs]
    setattr(klass, MAGIC_KEY, to_save)


def get_serialized_attributes(klass):
    """
    Given a class or instance, return a list of AttributeDescriptor instances
    for those attributes that should be serialized and deserialized.
    
    If set_serialized_attributes has been called on 'klass', it will determine
    the return value.
    
    Otherwise, 'klass' is first searched for matching pairs of getter/setter
    functions. If any are found, they are assumed to completely describe the
    serializable interface of the object.
    
    If neither of those checks finds anything, then all public data members are
    assumed to be serializable.
    """
    all_attrs = dir(klass)
    
    if MAGIC_KEY in all_attrs:
        to_serialize = getattr(klass, MAGIC_KEY)
    else:
        gs_pairs = get_paired_functions(klass)
        if gs_pairs:
            to_serialize = gs_pairs
        else:
            public_data = get_public_data_members(klass)
            to_serialize = [AttributeDescriptor(x) for x in public_data]
    
    # TODO: (sat 2/1/2011) This does not work as one might expect if
    # all attributes are defined only in the constructor for a class
    # (as is most common). We could try constructing an instance and
    # looking at its data members instead of looking at the class if
    # the class list is empty. But we need to think about this more to
    # see if it's sane, since RAII means that constructing a class is
    # not always without side effects.
    
    # To simplify human use and inspection, sort the list by name of
    # serialized object. It is expected that this is not expensive
    # relative to other operations being done in the context of call.
    to_serialize.sort(key=lambda x:x.name)
    return to_serialize


def get_paired_functions(klass, set_pat='set_', get_pat='get_str_'):
    """
    Look for pairs of functions in the given class or instance which appear to
    be getters and setters for the same attribute. Return a list of
    AttributeDescriptor instances describing the pairs. For example, with the
    default set_pat and get_pat, a class or instance containing functions
    
      set_foo
      get_str_foo
    
    would yield an AttributeDescriptor with:
    
      name: 'foo'
      setter_func_name: 'set_foo'
      getter_func_name: 'get_str_foo'
    
    Unpaired functions (just a setter or just a getter) are ignored. 
    """
    parts = []
    for p in (set_pat, get_pat):
        # A 'tag' here is just the 'foo' part of a function like set_foo
        tags = set(x.replace(p, '')
                   for x in dir(klass) if x.startswith(p))
        parts.append(tags)
    common_tags = parts[0].intersection(parts[1])

    ret = []
    for t in sorted(list(common_tags)):
        ad = AttributeDescriptor({'name': t,
                                  'setter_func_name': set_pat + t,
                                  'getter_func_name': get_pat + t,
                                  })
        ret.append(ad)  
    return ret
    
def get_public_data_members(klass):
    """
    Return a list of names of members of 'klass' (a class or instance) which
    appear to be public data, not methods or private.
    """
    all_attrs = dir(klass)
    public_attrs = [a for a in all_attrs if not a.startswith('_')]
    public_data = [a for a in public_attrs if not callable(getattr(klass, a))]
    return public_data

    
def compare_dicts(d1, d2):
    """
    Default dict comparison does the right thing for equality, but
    does not show you how dicts differ if they are not the same.
    
    This function returns a tuple of
    
    (differing_keys, first_subdict, second_subdict)
    
    differing_keys is essentially the set symmetric difference of keys in the
      two dicts
    first_subdict is the dict comprising those keys in differing_keys which are
      present in the first dict, and their values.
    second_subdict is the same for the second dict.  
    
    If the dicts are equal, the return value is ([], {}, {})
    """
    # TODO (sat 10-May-2011): This doesn't work for dicts with nested
    # dicts as values. set(d.items()) raises TypeError: dict objects
    # are unhashable. Need to recursively compare.
    differing_items = set(d1.items()).symmetric_difference(set(d2.items()))
    differing_keys = set(x[0] for x in differing_items)
    subdicts = [{}, {}]
    for k in differing_keys:
        for (idx, d) in enumerate([d1, d2]):
            if k in d.keys():
                subdicts[idx][k] = d[k]
    return [sorted(list(differing_keys))] + subdicts


def show(obj):
    from pprint import pprint
    pprint(instance_to_dict(obj))
