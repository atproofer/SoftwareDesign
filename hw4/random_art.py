# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:34:57 2014

@author: atproofer - mbocamazo
"""

# you do not have to use these particular modules, but they may help
from random import randint
from math import *          ## SS: I added this line - don't forget to add the math library dependency!
import Image

#need to include prod(a,b)=ab, cos_pi(a)=cos(pi*a), sin '', x(a,b)=a, y '' + 2 others
#avg, diff/2, sigmoid appropriate

full_list = ['x','y','prod','cos_pi','sin_pi','avg','half_diff','sigmoid']
unary_list = ['cos_pi','sin_pi','sigmoid']
binary_list = ['prod','avg','half_diff']
func_list = unary_list+binary_list
terminal_list = ['x','y']

def rand_selection(x):
    """Takes in list. Returns random element"""
    return x[randint(0,len(x)-1)]    

#The following two functions were explorations/visualizations of how to implement recursion 
    # and were not used
    
def unlimited_build():
    z = rand_selection(full_list)
    if z in terminal_list:
        return [z]
    if z in unary_list:
        return [z,unlimited_build]
    if z in binary_list:
        return [z,unlimited_build,unlimited_build]
        
def get_arg(case):
    if case==0:
        return [rand_selection(terminal_list)]
    if case==1:
        return [rand_selection(func_list)]
    if case==2:
        return [rand_selection(full_list)]   
    
def build_random_function(min_depth, max_depth):
    """Generates a random function based on the global function list and
    specified min and max depths. min_depth ought to be greater than max_depth."""
    #Depth numbering convention based on example provided in hw description
    
    #if depth is equal to maximum, take from the terminal list    
    if max_depth==1:
        return [rand_selection(terminal_list)]
          
    #if depth is less than minimum, take from the func list
    if min_depth>1:
        z = rand_selection(func_list)
        if z in unary_list:
            return [z,build_random_function(min_depth-1,max_depth-1)]
        if z in binary_list:
            return [z,build_random_function(min_depth-1,max_depth-1),build_random_function(min_depth-1,max_depth-1)]
    
    #if depth is less than maximum and greater or equal to minimum, take from the full list    
    if min_depth<=1 and max_depth>1:
        z = rand_selection(full_list)
        if z in terminal_list:
            return [z]
        if z in unary_list:
            return [z,build_random_function(min_depth-1,max_depth-1)]
        if z in binary_list:
            return [z,build_random_function(min_depth-1,max_depth-1),build_random_function(min_depth-1,max_depth-1)]
            

## SS: Passed my tests :)
## SS: for this function, you might make use of 'elif' statements, even though the functionailty
##     is the same, stylistically, it's preferable
def evaluate_random_function(f, x, y):
    """Takes as input a function constructed by nested lists, x, and y and returns 
    a value between 0 and 1 that evaluates it."""
    #print f[0] to check function as it is evaluated
    if f[0]=='x':
        return x
    if f[0]=='y':
        return y
    if f[0]=='prod':
        return evaluate_random_function(f[1], x, y)*evaluate_random_function(f[2], x, y)
    if f[0]=='cos_pi':
        return cos(pi*evaluate_random_function(f[1], x, y))
    if f[0]=='sin_pi':
        return sin(pi*evaluate_random_function(f[1], x, y))
    if f[0]=='avg':
        return (evaluate_random_function(f[1], x, y)+evaluate_random_function(f[2], x, y))/2
    if f[0]=='half_diff':
        return (evaluate_random_function(f[1], x, y)-evaluate_random_function(f[2], x, y))/2
    if f[0]=='sigmoid':
        return 1.0/(1.0+exp(-evaluate_random_function(f[1], x, y)))        
    print 'fail to identify' #if the argument isn't caught  
    print f[0]
    #unit test: ['prod', ['prod', ['x'], ['y']], ['prod', ['x'], ['y']]] should return (xy)^2
            
## SS: Passed my tests :)
## SS: maybe use some more descriptive variable names here
def remap_interval(val, input_interval_start, input_interval_end, output_interval_start, output_interval_end):
    """ Maps the input value that is in the interval [input_interval_start, input_interval_end]
        to the output interval [output_interval_start, output_interval_end].  The mapping
        is an affine one (i.e. output = input*c + b).
        Rescales and shifts the input value.    
    """
    c = float(output_interval_end-output_interval_start)/(input_interval_end-input_interval_start)
    b = output_interval_start
    return c*(val-input_interval_start)+b
    
def generate_image():
    """Generates an image using random functions methods. Saves output in folder."""
<<<<<<< HEAD
    for k in range(10):
        xsize = 350
        ysize = 350 
        im = Image.new("RGB",(xsize,ysize))
        pix = im.load()
        min_args = 12
            max_args = 18
        funcR=build_random_function(min_args,max_args)
        funcG=build_random_function(min_args,max_args)
        funcB=build_random_function(min_args,max_args)
        for i in range(xsize-1):
            xcoord = remap_interval(i,0,xsize,-1,1)         
            for j in range(ysize-1):
                ycoord = remap_interval(j,0,ysize,-1,1)            
                r=evaluate_random_function(funcR,xcoord,ycoord)
                g=evaluate_random_function(funcG,xcoord,ycoord)
                b=evaluate_random_function(funcB,xcoord,ycoord)
                mappedR = int(remap_interval(r,-1,1,0,255))
                mappedG = int(remap_interval(g,-1,1,0,255))
                mappedB = int(remap_interval(b,-1,1,0,255))
                pix[i,j]=(mappedR,mappedG,mappedB)
        name = 'ex4'+str(k)+'.png'
        im.save(name)                  
=======
    xsize = 350
    ysize = 350 
    im = Image.new("RGB",(xsize,ysize))
    pix = im.load()
    min_args = 4
    max_args = 5
    funcR=build_random_function(min_args,max_args)
    funcG=build_random_function(min_args,max_args)
    funcB=build_random_function(min_args,max_args)
    for i in range(xsize-1):
        xcoord = remap_interval(i,0,xsize,-1,1)         
        for j in range(ysize-1):
            ycoord = remap_interval(j,0,ysize,-1,1)            
            r=evaluate_random_function(funcR,xcoord,ycoord)
            g=evaluate_random_function(funcG,xcoord,ycoord)
            b=evaluate_random_function(funcB,xcoord,ycoord)
            mappedR = int(remap_interval(r,-1,1,0,255))
            mappedG = int(remap_interval(g,-1,1,0,255))
            mappedB = int(remap_interval(b,-1,1,0,255))
            pix[i,j]=(mappedR,mappedG,mappedB)
    im.save('ex7.png')            
            
generate_image()

## SS: Hey, great job! This is flawless in functionality (or so I can tell), and you have good
##     documentation, which I really appreciate. 



            
            
            
            
            
            
    
>>>>>>> 41504e1f25df6080449c122c4927de2d04be3089
