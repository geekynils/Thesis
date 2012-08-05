#!/usr/bin/python

import random
import os

from sys import argv

DEBUG = False

class Point:
    "Class representing a point in 2D."
    def __init__(self, x, y):
        self.x = x
        self.y = y

def intersect(p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):

    # Taken from http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect

    s1_x = p1_x - p0_x;         s1_y = p1_y - p0_y
    s2_x = p3_x - p2_x;         s2_y = p3_y - p2_y

    s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / (-s2_x * s1_y + s1_x * s2_y)
    t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / (-s2_x * s1_y + s1_x * s2_y)

    if s >= 0 and s <= 1 and t >= 0 and t <= 1:
        return True

    return False

def log_intersect(p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):
    print "Found intersection between (%2.2f %2.2f)-(%2.2f %2.2f) and (%2.2f %2.2f)-(%2.2f %2.2f)" \
          % (p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y)
        
def in_polygon(polygon_x, polygon_y, x, y):
    
    startx = 0
    starty = 0.001
    
    if len(polygon_x) != len(polygon_y):
        "Equal length of polygon_x and polygon_y required."
    
    i = 0
    num_intersect = 0
    
    # Iterate over all line segement of the polygon.
    while True:
        
        # Check for intersection of each line and the line starting at (0 0) and 
        # ending at (x y).
        if intersect(polygon_x[i], polygon_y[i], \
                     polygon_x[i+1], polygon_y[i+1], \
                     startx, starty, x, y):
            if DEBUG:
                log_intersect(polygon_x[i], polygon_y[i], \
                              polygon_x[i+1], polygon_y[i+1], \
                              startx, starty, x, y)
            
            num_intersect += 1
            
        i += 1
        
        if i == len(polygon_x)-2:
            
            # Last line from the last to the first point.
            l = len(polygon_x)
            if intersect(polygon_x[0], polygon_y[0], \
                         polygon_x[l-1], polygon_y[l-1], \
                         startx, starty, x, y):
                if DEBUG:
                    log_intersect(polygon_x[0], polygon_y[0], \
                                  polygon_x[l-1], polygon_y[l-1], \
                                  startx, starty, x, y)
                
                num_intersect +=1
            break
                
    
    if num_intersect % 2 == 0:
        return False
        
    return True

# i is the interval of the bounding box
def random_points(n, polygon_x, polygon_y, i1_x, i2_x, i1_y, i2_y):
    
    points = []
    x = 0.0
    y = 0.0
    
    while(len(points) < n):
        x = random.uniform(i1_x, i2_x)
        y = random.uniform(i1_y, i2_y)
        if in_polygon(polygon_x, polygon_y, x, y):
            points.append(Point(x, y))
    
    return points

def write_file(path, points):
    
    f = open(path, 'w')
    f.write(str(len(points)) + '\n')
    
    for point in points:
        f.write(str(point.x) + ' ' + str(point.y) + '\n')

    f.close()


# x coords in anti-clockwise order
polygon_x = [1, 3, 5.5, 3.75, 4.5, 2.5, 1.5]
polygon_y = [1, 0.5, 0.75, 2.5, 3.75, 4, 3.5]

# print "in_polygon(polygon_x, polygon_y, 3, 3) " \
#     + str(in_polygon(polygon_x, polygon_y, 3, 3))
# print "in_polygon(polygon_x, polygon_y, 1, 1) " \
#     + str(in_polygon(polygon_x, polygon_y, 1, 1))
# print "in_polygon(polygon_x, polygon_y, 5, 5) " \
#     + str(in_polygon(polygon_x, polygon_y, 5, 5))

if len(argv) == 3:
    n = argv[1]
    path = argv[2]
    points = random_points(int(n), polygon_x, polygon_y, 1, 5.5, 0.5, 4)
    write_file(path, points)
else:
    print "Usage: " + argv[0] + "[number of points] [file to store points]"
