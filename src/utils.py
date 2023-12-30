from math import log2

def int_to_binary_cors(integer, max_dimensions):
    cors = []
    if integer != 0:
        for i in range(int(log2(integer)+1)):
            cors.append(integer%2)
            integer = (integer)//2
    for _ in range(max_dimensions-len(cors)):
        cors.append(0)
    return cors
