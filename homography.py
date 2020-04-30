def point_transform(H, pt):
    """
    :param H: Homography matrix of dimension (3*3)
    :param pt: Point (x, y) to be transformed
    :return:
            return a transformed point ptrans = H*pt
    """

    a = H[0][0]*pt[0] + H[0][1]*pt[1] + H[0][2]
    b = H[1][0]*pt[0] + H[1][1]*pt[1] + H[1][2]
    c = H[2][0]*pt[0] + H[2][1]*pt[1] + H[2][2]
    ptrans = [a/c, b/c]
    return ptrans