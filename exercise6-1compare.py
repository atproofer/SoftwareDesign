# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 08:50:13 2014

@author: atproofer - MBocamazo
"""

#compare function exercise 6-1 of think python, MBocamazo
def compare(x,y):
    if x>y:
        return 1
    elif y>x:
        return -1
    else:
        return 0
print compare(1,2)
