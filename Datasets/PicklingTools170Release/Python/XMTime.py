"""
Python class that wraps the X-Midas time facilities.  The XMTime
is represented as two floats, one which contains the integral number
of seconds since the epoch, the other which contains the fractional number.

The following is always true of an XMTime:
self.integral == round(self.integral)
0.0 <= self.fractional < 1.0
"""

# $Id: xmtime.py 3442 2006-12-14 23:49:09Z dtj $

#import fixdlopenflags
#import xmpyapi

# Format "enumeration"
STANDARD = "_"
ACQ      = "A"
EPOCH    = "E"
NORAD    = "N"
TCR      = "T"
VAX      = "V"

# Warning level "enumeration"
IGNORE    = 0
WARNING   = 1
EXCEPTION = 2

class XMTimeInvalidTC_PREC(Exception):
    def __init__(self, hdr):
        Exception.__init__(self,
                           'TC_PREC main keyword in file %s does not '
                           'evaluate to a float: %s' %
                           (hdr.get('file_name', ''),
                            repr(hdr['keywords']['TC_PREC'])))


class XMTime(object):
    # Slots make construction and look-up of attributes in a Python
    # class more efficient and catches attempts to assign to
    # misspelled attribute references (e.g. XMTime.ingetral = ...)
    __slots__ = ('integral', 'fractional')

    # __setstate__/__getstate__ are required for pickling/copying when
    # __slots__ is defined. To maintain pre-__slots__ backwards
    # compatibility use the former __dict__ contents as the state.
    def __getstate__(self):
        return dict(integral=self.integral, fractional=self.fractional)
    def __setstate__(self, state):
        self.integral = state['integral']
        self.fractional = state['fractional']

    # __reduce__ introduces a more compact and efficient pickle/copy
    # method for post-__slots__ pickled XMTimes.  The __get/setstate__
    # methods will only be called for pre-__slots__ pickled XMTimes.
    def __reduce__(self):
        """helper for pickle"""
        return (XMTime, self.two_floats())

    def __init__(self, *args):
        """
        Create an XMTime.

        If no arguments are given, create an XMTime representing the
        current time of day.

        XMTime(seconds): If one float is given, assume it is a number of
        seconds, and create a time from it relative to J1950.

        XMTime(integral, fractional): If two floats are given, assume
        it is the integral and fractional parts of an XMTime, and
        construct this XMTime directly from that.

        XMTime(string [,format] [,warning level]):  If a string is given,
        convert it to a time according to the formats documented in the
        help given by xm.help('time_format').  If the 'format' argument
        is not supplied, the standard format is assumed.  The optional
        'warning level' argument allows the user to specify the desired
        behavior if an invalid time string is given.  Allowable warning
        levels are as follows:

                    0   IGNORE      Warnings are suppressed
        Default --> 1   WARNING     A warning is printed to the screen
                    2   EXCEPTION   An exception is thrown

        If 'warning level' is omitted or is a value other than 0, 1, or
        2, it defaults to warning level 1.

        XMTime(hcb): If a dictionary is given, assume it is a
        dictionary representation of a BLUE header, such as that
        returned by bluefile.readheader(), and extract any precision
        timecode contained within it. Note that the header's 'xstart'
        field is not taken into account, for symmetry with the X-Midas
        STATUS command. To get the time of the first sample in the
        file described by 'hcb', use this expression:

          XMTime(hcb) + hcb['xstart']

        If the hcb['keywords']['TC_PREC'] main header keyword is present
        and it is not a string (all main header keywords default to a
        string type) or a float an XMTimeInvalidTC_PREC exception is
        raised.  If a string is present but it cannot be converted to
        a float via the float() the standard ValueError exception will
        be raised.
        """
        if len(args) == 0:
            self.integral, self.fractional = xmpyapi.now()
        else:
            if isinstance(args[0], str):
                # string, optional format, optional warn flag
                time_str = args[0]

                # Default values
                warn = WARNING
                fmt  = STANDARD

                if len(args) == 2:
                    # User entered time string and one optional argument
                    if isinstance(args[1], str):
                        # Optional argument must be the format
                        fmt = args[1]
                    else:
                        # Optional argument must be the warning level
                        warn = args[1]
                elif len(args) >= 3:
                    # User supplied all arguments
                    fmt  = args[1]
                    warn = args[2]

                if warn not in [IGNORE, WARNING, EXCEPTION]:
                    warn = WARNING
                    
                junk, self.integral, self.fractional = \
                      xmpyapi.time_format(time_str, fmt.upper(), warn)

            elif isinstance(args[0], XMTime):
                self.integral, self.fractional = \
                               args[0].integral, args[0].fractional
            elif isinstance(args[0], dict) and args[0].has_key('timecode'):
                # Assume a header from bluefile
                hdr = args[0]
                tc = hdr['timecode']
                tc_prec = hdr.get('keywords', {}).get('TC_PREC', 0.0)
                if isinstance(tc_prec, str):
                    tc_prec = float(tc_prec)
                elif not isinstance(tc_prec, float):
                    raise XMTimeInvalidTC_PREC(hdr)
                self.integral = round(tc)
                # We're getting this from hcb.timecode, so we know
                # that it has precisely 6 digits after the decimal.
                self.fractional = round((tc - self.integral), 6) + tc_prec
            else:
                if len(args) == 1:
                    t = args[0]
                    tint = round(t)
                    self.integral, self.fractional = tint, t - tint
                else:
                    self.integral, self.fractional = args
        self._normalize()

    def _fracstr(self):
        """
        Return the fractional part of the time as a string. Used to be
        done as '%.12f', but Python's string conversion uses the
        rint() function, not Fortran NINT() as X-Midas does. The
        underlying X-Midas time conversion utilities are used for
        consistency.  _fracstr is called by hash() and cmp(), so that
        times that look the same compare and hash the same.
        """
        # times2tod() is used in order to avoid involving the date
        # conversion utilities.
        s = xmpyapi.times2tod(0, self.fractional)
        ii = s.find('.')
        if ii >= 0:
            return s[ii:]
        else:
            return ''

    def put_epoch(self, hcb):
        """
        Given a dictionary representation of a BLUE header (such
        as that returned from bluefile.readheader), fill in its
        'timecode' field and 'TC_PREC' main keyword according
        to the BLUE timecode standard.

        The inverse of this function is accomplished by constructing
        an XMTime from a dictionary rep. of a BLUE header.

        The value of the 'TC_PREC' keyword added to the dictionary
        passed in will be a string representation of a float value;
        all main header keywords can only have string values.
        """
        timeu = round(self.fractional, 6)
        timep = round((self.fractional - timeu), 12)
        hcb['timecode'] = self.integral + timeu
        if timep != 0.0:
            if not hcb.has_key('keywords') or hcb['keywords'] is None:
                hcb['keywords'] = {}
            hcb['keywords']['TC_PREC'] = str(timep)
        else:
            try:
                del(hcb['keywords']['TC_PREC'])
            except:
                pass

    def eqt(self, other_time, tol):
        """
        Return whether the given XMTime is equal to this XMTime
        within some tolerance (i.e. is abs(difference) <= tol).
        """
        diff = self.__sub__(other_time)
        return abs(diff.seconds()) <= tol

    def two_floats(self):
        """
        Return the implementation of this XMTime, that is, two
        floating point numbers representing the integral and
        fractional number of seconds, respectively, past the
        J1950 epoch.
        """
        return self.integral, self.fractional

    def seconds(self):
        """
        Add the integral and fractional number of seconds to
        return a single floating point value representing the
        whole time.  Some precision will be lost in the result;
        only the time up to about the microsecond level will be
        accurate.
        """
        return self.integral + self.fractional

    def date(self):
        """
        Return the date portion of an XMTime.
        """
        return xmpyapi.times2str(self.integral, 0.0, '_DATE')

    def hms(self):
        """
        Return the time portion of an XMTime, in hms format
        (hh:mm:ss).
        """
        return str(self).split('::')[-1].split('.')[0]

    def round(self, digits=0):
        """
        Round an XMTime to a given precision in decimal digits
        (default 0 digits) and return as an XMTime. Precision may be
        negative and is applied to the full J1950 seconds.

        The built-in Python round() method can round XMTimes
        incorrectly when digits >= 0. Internally the XMTime is
        converted to a float before rounding, i.e. (XMTime.seconds()),
        and in doing so the full precision required to round
        accurately is lost.  Furthermore, the returned float value
        converted back to an XMTime often contains sub-microsecond
        floating point noise.

        Note: J1950 seconds are rounded to match the behavior of
        Python's built-in round() method. Hence, when digits is < -1
        (i.e. 100's, 1000's, etc.) the resulting XMTime will not
        necessarily fall on minute, hour, day, etc. boundaries.
        """
        if digits < 0:
            # Rounding boundaries are at integral values (5, 50, 500, etc)
            # hence the fractional portion is irrelevant.
            return XMTime(round(self.integral, digits))
        else:
            # If the fractional portion rounds up, the XMTime constructor
            # will normalize it.
            return XMTime(self.integral, round(self.fractional, digits))

    def __float__(self):
        """
        Same as self.seconds().
        """
        return self.seconds()

    def __str__(self):
        """
        Create a standard string representation of this XMTime.
        """
        return self.asString()

    def __sub__(self, t):
        """
        Subtract a float or another XMTime from this XMTime.
        The result is another XMTime.
        """
        if not isinstance(t, XMTime):
            t = XMTime(t)
        temp = XMTime(self)
        temp.integral -= t.integral
        temp.fractional -= t.fractional
        temp._normalize()
        return temp

    def __add__(self, t):
        """
        Add a floating point number or another XMTime to an XMTime to
        produce a new XMTime.
        """
        temp = XMTime(self)
        if isinstance(t, XMTime):
            temp.integral += t.integral
            temp.fractional += t.fractional
        else:
            tint = round(t)
            temp.integral += tint
            temp.fractional += (t - tint)
        temp._normalize()
        return temp

    def __radd__(self, t):
        """
        Add a floating point number or another XMTime to an XMTime to
        produce a new XMTime.
        """
        return self.__add__(t)

    def asString(self, format=STANDARD):
        """
        Return a string representation of this time, according to
        the given format.
        """
        return xmpyapi.times2str(self.integral, self.fractional, format)

    def __repr__(self):
        """
        Return the standard string representation of an XMTime object,
        in Python-object syntax:  <xmtime.XMTime YYYY:MM:DD::hh::mm::ss.ssss>
        """
        return "<xmtime.XMTime %s>" % str(self)

    def __hash__(self):
        """
        Return a unique hash value for this instance that is determined
        by its value.
        """
        # Cannot use two_floats(), because the floats may
        # have accumulated sub-picosecond precision, and yet XMTime is
        # only intended to be exact to picoseconds. Sub-picosecond precision
        # is not stored in BLUE headers, is not written out in the
        # str() method, and so two XMTimes that differ by less than a
        # picosecond should be considered the same for this function.
        return hash((self.integral, self._fracstr()))

    # ##### Relational operators

    def __cmp__(self, t):
        """
        XMTimes are only intended to record time to the nearest picosecond.
        Common sense suggests that times that have the same string
        representation will compare as equal, as well as the converse.
        Thus, when times are within a second of each other, only the
        first twelve digits to the right of the decimal, with rounding
        from the thirteenth, are considered in the comparison.
        """
	tt = XMTime(t)
        icmp = cmp(self.integral, tt.integral)
        if icmp:
            return icmp
        else:
            return cmp(self._fracstr(), tt._fracstr())

    def _normalize(self):
        """
        Makes sure that self.integral has all of its fractional parts
        moved into self.fractional, and that self.fractional has all
        of its integral parts moved into self.integral.
        Ensure that self.fractional is not storing
        more than picoseconds.
        Also make sure that self.fractional is positive.
        """
        iint = round(self.integral)
        self.fractional = round(self.fractional, 12)
        if iint != self.integral:
            self.fractional += (self.integral - iint)
            self.integral = iint
        if abs(self.fractional) >= 1.0:
            ifrac = round(self.fractional)
            self.integral += ifrac
            self.fractional -= ifrac
        assert round(self.integral) == self.integral
        assert abs(self.fractional) < 1.0
        if self.fractional < 0.0:
            self.fractional += 1
            self.integral -= 1
        assert 0.0 <= self.fractional < 1.0




# The following class is a replacement for the methods from the xmpyapi
# module the XMPY xmtime module relies on.
from struct import calcsize as _calcsize

class xmpyapi(object):
    # The start of the Unix 1970 epoch, expressed in terms of the
    # J1950 epoch.  To be added to time.time() in order to obtain the
    # right value, or to be subtracted from self.integral before using
    # Unix time utilities for string time representation.
    EPOCH_DIFF = 631152000.0
    # time.strftime() overflows on 32 bit machines past this time
    LONG_IS_32_BIT = (_calcsize('l') == 4)
    UNIX_32BIT_EPOCH_MAX = 2777068800.0

    SECS_PER_MIN = 60.0
    SECS_PER_HOUR = 3600.0
    SECS_PER_DAY = 86400.0
    SECS_PER_NON_LEAP_YEAR = 31536000.0 # J1950 is a not a leap year
    NCOLONS_MAP = {
        0 : [1],                                            # ss.sss
        1 : [SECS_PER_HOUR, SECS_PER_MIN],                  # hh:mm
        2 : [SECS_PER_HOUR, SECS_PER_MIN, 1],               # hh:mm:ss
    }

    # _parse_date() assumes that any instance of '%y/%Y' are at the
    # beginning of the format string to handle overflow/underflow
    # problems as well as date wrapping around 1950 vs 1970 for 2
    # digit years. Currently VAX is out-of-luck for under/overflow
    # issues.
    TFORMAT = { STANDARD:('%Y:%m:%d::%H:%M:%S', 'yyyy:mm:dd::hh:mm:ss.sss'),
                     VAX:('%d-%b-%Y:%H:%M:%S',  'dd-MON-yyyy::hh:mm:ss.sss'),
                   EPOCH:('%Y:',                'yyyy:sec_of_year.ssss'),
                   NORAD:('%y%j',               'yyDOY:frac_day'),
                     TCR:('%j:%H:%M:%S',        'DOY:hh:mm:ss.sss'),
                     ACQ:('%y.%j:%H:%M:%S',     'yy.DOY:hh:mm:ss.sss'),
                  '_DATE':('%Y:%m:%d',           'yyyy:mm:dd')} 

    @staticmethod
    def now():
        import time, math
        fractional, integral = math.modf(time.time())
        # System time does not have more than microsecond precision.
        # Drop the bits that suggest otherwise
        return integral + xmpyapi.EPOCH_DIFF, round(fractional, 6)

    @staticmethod
    def time_format(time_str, fmt, warn):
        try:
            integral, fractional = xmpyapi._parse_time(time_str, fmt)
        except Exception, e:
            if warn == EXCEPTION:
                raise
            elif warn == WARNING:
                print 'WARNING:', e
            integral, fractional = 0.0, 0.0
        return None, integral, fractional

    @staticmethod
    def _parse_time(time_str, fmt):
        from math import floor
        # Pull off the fractional part separately
        integral = 0.0
        idec = time_str.rfind('.')
        if idec >= 0 and (fmt != ACQ or idec > time_str.rfind(':')):
            fractional = float(time_str[idec:])
            time_whole = time_str[:idec].strip()
        else:
            time_whole = time_str.strip()
            fractional = 0.0
        
        time_fmt = xmpyapi.TFORMAT.get(fmt, (None, None))[0]
        if not time_fmt:
            raise Exception('XMTime format "%s" not yet implemented' % fmt)

        if fmt == STANDARD:
            # We need to be much more forgiving than time.strptime() will
            # be below, so we parse out the HH:MM:SS ourselves below
            date_time = time_whole.split('::')
            if len(date_time) > 2 or not date_time[0]:
                raise XMTimeParseError(fmt, time_str)
            if date_time[-1]:
                # Manually compute time string integral, map empty strings
                # to 0 Either SS, HH:MM or HH:MM:SS
                vals = [int(_ or 0) for _ in date_time[-1].split(':')]
                if len(vals) > 3:
                    raise XMTimeParseError(fmt, time_str)
                if len(vals) == 1:
                    integral = float(vals[0]) # SS
                elif len(vals) > 1:
                    # HH:MM[:SS]
                    integral = (float(vals[0]) * xmpyapi.SECS_PER_HOUR +
                                float(vals[1]) * xmpyapi.SECS_PER_MIN)
                    if len(vals) > 2:
                        integral += vals[2] # [:SS]
            if len(date_time) == 1:
                # No year, we're done
                return integral, fractional
            elif len(date_time[0].split(':')) == 1:
                # Date portion has no ':' separators, interpret as offset
                # days (i.e. a duration)
                integral += float(date_time[0]) * xmpyapi.SECS_PER_DAY
                return integral, fractional
            # Just the YEAR part needs to be parsed
            time_whole = date_time[0]
            time_fmt = time_fmt.split('::')[0]
        elif fmt == NORAD:
            # Fractional part is fractional day
            fractional *= xmpyapi.SECS_PER_DAY
            integral = floor(fractional)
            fractional -= integral
        elif fmt == EPOCH:
            # %Y:%<soy> is second of year. time.strptime() does not handle
            # second-of-year, so unstringize this part ourselves.
            parts = time_whole.split(':')
            if len(parts) != 2:
                raise XMTimeParseError(fmt, time_str)
            time_whole, integral = parts[0] + ':', float(parts[1])
        if time_fmt:
            try:
                integral += xmpyapi._parse_date(time_whole, fmt, time_fmt)
            except:
                raise XMTimeParseError(fmt, time_str)
        return integral, fractional

    @staticmethod
    def _parse_date(time_whole, fmt, time_fmt):
        import time, sys
        # Account for the J1950 -> J1970 epoch diff
        integral = xmpyapi.EPOCH_DIFF
        if time_fmt.startswith('%y'):
            # time.strptime() interprets splits 19xx/20xx 2 digit
            # years at 1969,  xmtime.XMTime splits them at 1950.
            yy = int(time_whole[:2])
            if yy < 69:
                time_fmt = '%Y' + time_fmt[2:]
                if yy > 49:
                    time_whole = '19' + time_whole
                elif sys.maxint < 4e9 and yy > 37:
                    # On 32 bit systems, time.mktime() wll
                    # overflow if not 1901 < year < 2038, so we
                    # need to do some extra magic knowing the
                    # EPOCH_DIFF between 2030 and 2050 is the same
                    # as between 1950 and 1970
                    integral += xmpyapi.EPOCH_DIFF
                    time_whole = str(2000+yy-20) + time_whole[2:]
                else:
                    time_whole = '20' + time_whole
        elif time_fmt.startswith('%Y') and sys.maxint < 4e9:
            yy = int(time_whole[:4])
            while yy > 2037:
                integral += xmpyapi.EPOCH_DIFF
                yy -= 20
            while yy < 1902:
                integral -= xmpyapi.EPOCH_DIFF
                yy += 20
            time_whole = str(yy) + time_whole[4:]

        unix_time = time.strptime(time_whole, time_fmt)
        # In TCR, the year is always the current year
        if fmt == TCR:
            unix_time = (time.gmtime()[0],) + unix_time[1:]
        return integral + time.mktime(unix_time)

    @staticmethod
    def times2str(integral, fractional, format):
        import time
        fmt = xmpyapi.TFORMAT.get(format, (None, None))[0]
        if not fmt:
            raise Exception("Unhandled time format: " + format)

        if (format == STANDARD and
            abs(integral) < xmpyapi.SECS_PER_NON_LEAP_YEAR):
            # If integral is within a year of the J1950 epoch
            # year then assume it's a duration. We are taking
            # advantage of the fact here that the UNIX epoch year
            # 1970 is identical in number of days to J1950.
            unix_time = time.gmtime(integral)
            istr = time.strftime('%H:%M:%S', unix_time)
            doy = unix_time.tm_yday - 1
            if integral < 0: doy -= 365
            if doy: istr = '%d::%s' % (doy, istr)
        else:
            import sys
            unix_epoch = integral - xmpyapi.EPOCH_DIFF
            if abs(unix_epoch) > sys.maxint and fmt.upper().find('%Y') >= 0:
                # Handle 32 bit overflow by calculating at an
                # equivalent year and manually replacing %Y and %y in
                # the fmt string before calling time.strftime().
                #
                # xmtime.XMTime does not properly handle leap year at
                # centuries that are not divisible by 400 (i.e. 1900,
                # 2100) so we don't handle it here either.
                eoff = int(unix_epoch / xmpyapi.EPOCH_DIFF)
                unix_time = time.gmtime(unix_epoch - eoff * xmpyapi.EPOCH_DIFF)
                year = str(unix_time[0] + (eoff * 20))
                fmt = fmt.replace('%Y', year).replace('%y', year[2:])
            else:
                unix_time = time.gmtime(unix_epoch)
            istr = time.strftime(fmt, unix_time)

        if format == NORAD:
            # Now generate <frac_of_day> by combining the <sec_of_day>
            # and the fractional seconds.
            fractional += (integral % xmpyapi.SECS_PER_DAY)
            fractional /= xmpyapi.SECS_PER_DAY
            if fractional:
                # Get as much precision as possible since we can only keep
                # up to .1 nanosecond precision near the end of the day.
                ifrac = ('%.16f' % fractional).rstrip('0')
                if ifrac.startswith('1.'):
                    # Too close... we rounded up, can't change the integral
                    # part, so indicate just below 1.0
                    return istr + '.9999999999999999'
                elif not ifrac.endswith('.'):
                    return istr + ifrac[1:]
            return istr

        elif format == EPOCH:
            # %Y:%<soy> is second of year. time.strftime() does not handle
            # second-of-year, so stringize this part ourselves.
            istr += str(int(integral - float(XMTime(istr + "01:01::"))))

        ifrac = xmpyapi.times2tod(0, fractional)
        if ifrac == '0':
            return istr
        elif ifrac.endswith('000000'):
            # To match X-Midas, show sub-microsecond only when present
            return istr + ifrac[1:-6]
        else:
            return istr + ifrac[1:]

    @staticmethod
    def times2tod(unused, fractional):
        # This method is called by the XMPY xmtime module only to hash
        # and compare XMTime instances. The underlying X-Midas routine
        # called uses Fortran NINT() for rounding the fractional
        # seconds which rounds up at .5. The '%.12f' string conversion
        # uses the rint() (as does printf()) which rounds 0.5 down. To
        # keep the comparisons consistent with the Fortran, they
        # called into X-Midas. However, Python's round(val, 12)
        # function also gives the desired NINT rounding behavior
        # without needing to call into X-Midas.

        # According to the original author of xmtime.XMTime, the
        # resulting rounded value is converted to a string because it
        # was found the sub-picosecond bits of the rounded value were
        # not always consistent. This would cause unpredictable
        # behavior when hashing or comparing two XMTimes that should
        # otherwise be equal. When this port of X-Midas-free XMTime
        # was written in 11/2009, this behavior could not be
        # reproduced using round(fractional, 12) in limited
        # testing. It is possible this may now be enough and no
        # conversion to string is necessary, in which case the need
        # for the XMTime._fracstr() method goes away all-together.
        v = round(fractional, 12)
        if v:
            return '%.12f' % v
        else:
            return '0'


class XMTimeParseError(Exception):
    def __init__(self, fmt, got):
        Exception.__init__(self,
                           '"%s" time expected %s, got "%s". Failed to create '
                           'time formatted string.' %
                           (fmt, xmpyapi.TFORMAT[fmt][-1], got))
