def simple_hash(s, base):
    cumsum= 0
    for i in range(1, len(s)+1):
        cumsum+= base**(len(s)-i) * ord(s[i])
    return cumsum
