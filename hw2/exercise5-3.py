# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 08:30:21 2014

@author: atproofer - MBocamazo
"""
#softdes exercise 5-3 of thinkpython, check fermat
def prompt_fermat():
    a = int(raw_input("What is the value of a?\n"))
    b = int(raw_input("What is the value of b?\n"))
    c = int(raw_input("What is the value of c?\n"))
    n = int(raw_input("What is the value of n?\n"))
    check_fermat(a,b,c,n)
def check_fermat(a,b,c,n):
    if n>2&(a^n+b^n)==c^n:
        print "Holy smokes, Fermat was wrong!"
    else:
        print "No, that doesn’t work."
        
prompt_fermat()
        
    
