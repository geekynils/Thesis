#!/usr/bin/python

import sys

filename = sys.argv[1]

f = open(filename, 'r')
lines = f.readlines()

i = 0
avg = []

while i < len(lines)/10:
    k = 0
    sum = 0
    while k < 10:
        sum += int(lines[i * 10 + k])
        k += 1
    avg.append(sum / 10)
    i+=1
    
for i in avg:
    print i