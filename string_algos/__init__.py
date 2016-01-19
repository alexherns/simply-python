from python_tools import tools

def check_args(bigstr, littlestr):
    assert isinstance(bigstr, str) and isinstance(littlestr, str),\
        """Please supply objects of string type"""
    assert len(bigstr) >= len(littlestr),\
        """bigstr must be at least as long as littlestr"""

def naive_matching(bigstr, littlestr):
    check_args(bigstr, littlestr)
    match_sites= []
    m= len(littlestr)
    for start in range(len(bigstr)-m+1):
        if bigstr[start:start+m] == littlestr:
            match_sites.append(start)
    return match_sites

def rabin_karp_matching(bigstr, littlestr, base=64):
    check_args(bigstr, littlestr)
    match_sites= []
    littlehash= tools.simple_hash(littlestr, base=base)
    m= len(littlestr)
    bighash= tools.simple_hash(bigstr[:m], base=base)
    for start in range(len(bigstr)-m+1):
        if bighash == littlehash:
            match_sites.append(start)
        bighash= base*(bighash-base**(m-1))+ord(bigstr[start+m])
    return match_sites

