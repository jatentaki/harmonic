def rot90(x, k=1, plane=(4, 5)):
    if k == 0:
        return x
    elif k == 1:
        return x.flip(plane[0]).transpose(*plane)
    elif k == 2:
        return x.flip(plane[0]).flip(plane[1])
    elif k == 3:
        return x.flip(plane[1]).transpose(*plane)
    else:
        raise ValueError("k={} is invalid".format(k))
