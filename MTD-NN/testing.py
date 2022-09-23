import math

#relevant variables
rounding = False
sf = 3
T = 15
p = 70

#find b and round
def b(T):
    b = (-246+3.5*T-0.03*(T**2))
    if rounding: b = round(b, sf - int(math.floor(math.log10(abs(b)))) - 1) 
    b = b*(10**-5)
    return b

#find c and round
def c(T):
    c = (120+1.2*T+0.093*(T**2))
    if rounding: c = round(c, sf - int(math.floor(math.log10(abs(c)))) - 1)
    c = c*(10**-8)
    return c

#find compressibility factor
def z(T,p):
    z = 1 + b(T)*p + c(T)*(p**2)
    return z

print(b(T))
print(c(T))
print(z(T,p))