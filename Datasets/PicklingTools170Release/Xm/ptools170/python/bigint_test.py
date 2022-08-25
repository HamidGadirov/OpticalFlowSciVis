
# Simple timing test to see how fast multiplies, adds, divides, subtracts
# are for long integers

def nk(n,k) :
   sym = n-k
   if sym>0 and sym<k : k = sym
   
   num = 1
   den = 1
   for ii in xrange(0,k):
      num *= n-ii
      den *= ii+1
   return num/den

sum=0
for n in xrange(1000, 1099) :
  for x in xrange(0, n) :
      sum +=  nk(n, x)
print sum

