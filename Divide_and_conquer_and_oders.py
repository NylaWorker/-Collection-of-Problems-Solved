'''
Author: Nyla Worker
Following the sudo code from class.

'''

import sys


def closest_pair(xy_list): # sorts x once and then calls recursive funtion.
    '''
    :param xy_list: a list of tupple pairs (x,y)
    :return: two tuples and the distance between them, which is the smallest across all points.
    '''
    if len(xy_list)>0: #checking it is not empty
        print(xy_list)
        # presorting pairs by x and y to not recompute withing the functions.
        # this is space inneficient.
        xy_xlist = sorted(xy_list, key=lambda k: [k[0]])  # sorts by x
        xy_ylist =sorted(xy_list,key=lambda k: [k[1]])
        print()
        return  cp(xy_xlist,xy_ylist)


def dist(xy1,xy2):
    x1,y1,=xy1
    x2,y2 =xy2
    return ((x1-x2)**2+(y1-y2)**2)**(1/2)


def brute(xy_list):
    '''
    :param xy_list: List of tree points
    :return: smallest distance
    '''
    d1 = dist(xy_list[0], xy_list[1])
    if len(xy_list)==2:
        return xy_list[0], xy_list[1],d1
    d1 = dist(xy_list[0],xy_list[1])
    d2 = dist(xy_list[0],xy_list[2])
    d3 = dist(xy_list[1],xy_list[2])
    if d1<=d2 and d1<=d3:
        return xy_list[0],xy_list[1], d1
    elif d3<=d1 and d3<=d2:
        return xy_list[1],xy_list[2],d3
    else:
        return  xy_list[0],xy_list[2],d2

def cp(xy_xlist, xy_ylist):
    '''

    :param xy_xlist: list of (x,y) tuples ordered by x
    :param xy_ylist: list of (x,y) tuples ordered by y
    :return: two tuples and the distance between them, which is the smallest across all points.
    '''

    l = len(xy_xlist)
    #base case: Brute force
    if l<=3:
        return brute(xy_xlist)
    #dividing
    mid = len(xy_xlist)//2
    xl = xy_xlist[:mid]
    xr = xy_xlist[mid:]
    midx = xy_xlist[mid][0]

    yl,yr = list(),list()
    for x in xy_ylist:
        if x[0] <=midx:
            yl.append(x)
        else:
            yr.append(x)
    #conquering
    pl, ql, minl = cp(xl,yl) # divides into the left half
    pr, qr, minr = cp(xr, yr) # divides into the right half

    if minl< minr:
        delta = minl
        mp = (pl, ql)
    else:
        delta = minr
        mp = (pr, qr)

    #merging
    pm,qm, minm = closest_split(xy_xlist,xy_ylist,delta, mp)

    if delta<=minm:
        return mp[0],mp[1],delta
    else:
        return pm,qm, minm


def closest_split(xy_xlist,xy_ylist,delta, mp):
    '''

    :param xy_xlist:  list of (x,y) tuples ordered by x
    :param xy_ylist: list of (x,y) tuples ordered by y
    :param delta: smallest distance from divided segments
    :param mp: two (x,y) tuples with smallest distance
    :return: two tuples and the distance between them, which is the smallest across all points.
    '''
    lnx = len(xy_xlist)
    mx = xy_xlist[lnx//2][0]
    #Array of all the points within the delta bound

    sy = [x for x in xy_ylist if mx -delta<x[0]<=mx +delta]
    if len(sy)==1: #avoiding unnecessary computation but not really needed.
        return mp,delta
    bestmin =delta
    lensy = len(sy)
    best_pair = mp
    for i in range(lensy -1):
        for j in range(i+1, min(i+7,lensy)): #we showed in class how we only need to go
            p,q = sy[i], sy[j] #thorugh the first couple of squares.
            dst = dist(p,q)
            if dst < bestmin:
                best_pair =p,q
                bestmin = dst
    return best_pair[0],best_pair[1],bestmin



def main():
    data = sys.stdin.readlines()
    tests = []
    cur=[]
    for line in data:
        line.split(" ")
        if len(line) == 1:
            tests.append(cur)
            cur = []
        cur.append((line[0],line[1]))
    tests.pop(0)
    for test in tests:
        closest_pair(test)


main()


#  PERSONAL TESTS
#     # Now we need to find all the x values within that bound and sort the ys
#
# import random # Testing this. This generates random points.
# def test_case(length: int = 10000):
#     xs = [random.randint(-length, length) for i in range(length)]
#     ys = [random.randint(-length, length) for i in range(length)]
#     return xs, ys
#
# x,y= test_case(4)
# print(closest_pair(list(zip(x, y))))
