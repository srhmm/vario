import math

def cantor_pairing(x, y):
    return int((x + y) * (x + y + 1) / 2 + y)

def logg (val):
    """Log 0 = 0 convention in mdl scores"""
    if(val==0): return 0
    else: return math.log(val)