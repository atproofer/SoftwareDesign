# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 08:09:48 2014

@author: atproofer - MBocamazo
"""
#p is the number of rows, q is the number of columns
def printnbyn(p,q):
    k = 'x '+'- '*4
    j = '| '+'  '*4
    x = k*q+'x'
    y = j*q+'|'
    for m in range(0,p):
        print x
        for i in range(0,4):
            print y
    print x
printnbyn(2,2)