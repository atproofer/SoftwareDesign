# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 11:24:42 2014

@author: Michael Bocamazo
"""

# you may find it useful to import these variables (although you are not required to use them)
from amino_acids import aa, codons
from random import random

def collapse(L):
    """ Converts a list of strings to a string by concatenating all elements of the list """
    output = ""
    for s in L:
        output = output + s
    return output

def get_complement_strand(dna):
    """Converts a strand string to its complement strand"""
    x = ''
    for i in range(len(dna)): #Assumes proper bases in input
        if dna[i]=='A':
            x=x+'T'
        elif dna[i]=='T':
            x=x+'A'
        elif dna[i]=='C':
            x=x+'G'
        elif dna[i]=='U':#Accounting for a possible Uracil
            x=x+'A' 
        else:
            x=x+'C'
    return x
    
def get_complement_strand_unit_tests():
    """ Unit tests for the get_complement_strand function """
    print "input: 'ATGCGA'"
    print "expected output:TACGCT"
    print "actual output:  "+get_complement_strand('ATGCGA')
    print "\ninput: 'TTA'"
    print "expected output:AAT"
    print "actual output:  "+get_complement_strand('TTA')
    print "\ninput: 'ATGCCCGCTTT'"
    print "expected output:TACGGGCGAAA"
    print "actual output:  "+get_complement_strand('ATGCCCGCTTT')
    print "\ninput: 'TCTACU'" #Adding a test for Uracil
    print "expected output:AGATGA"
    print "actual output:  "+get_complement_strand('TCTACU')
        

def coding_strand_to_AA(dna):        
    """ Computes the Protein encoded by a sequence of DNA.  This function
        does not check for start and stop codons (it assumes that the input
        DNA sequence represents an protein coding region).
        
        dna: a DNA sequence represented as a string
        returns: a string containing the sequence of amino acids encoded by the
                 the input DNA fragment                 
    """
    """note that the inverse operation is impossible because the function is surjective"""
    x = '' #initialize return string
    for i in range((len(dna)/3)):
        snippet = dna[3*i:3*i+3] #takes a codon-length part
        for j in range((len(codons))):
            if snippet in codons[j]: #checks if the snippet is in the nested list
                x=x+aa[j] #when equality is found, adds amino acid name to the list
    return x   
    
def coding_strand_to_AA_unit_tests():
    """ Unit tests for the coding_strand_to_AA function """
    print "input: 'ATGCGA'"
    print "expected output:MR"
    print "actual output:"+coding_strand_to_AA('ATGCGA')
    print "\ninput: 'TTA'"
    print "expected output:L"
    print "actual output:"+coding_strand_to_AA('TTA')
    print "\ninput: 'ATGCCCGCTTT'"
    print "expected output:MPA"
    print "actual output:"+coding_strand_to_AA('ATGCCCGCTTT')
    print "\ninput: 'TCTACT'"
    print "expected output:ST"
    print "actual output:"+coding_strand_to_AA('TCTACT')

def get_reverse_complement(dna):
    """ Computes the reverse complementary sequence of DNA for the specfied DNA
        sequence
    
        dna: a DNA sequence represented as a string
        returns: the reverse complementary DNA sequence represented as a string
    """
    x = ''
    for i in range(len(dna)):
        x = x+dna[len(dna)-i-1] #reverses list
    return get_complement_strand(x) #returns the complement  
    
def get_reverse_complement_unit_tests():
    """ Unit tests for the get_complement function """
    print "input: 'TCGACTGAC'" #Changing two of the unit tests
    print "expected output:GTCAGTCGA"
    print "actual output:  "+get_reverse_complement('TCGACTGAC')
    print "\ninput: 'ATGC'"
    print "expected output:GCAT"
    print "actual output:  "+get_reverse_complement('ATGC')
    print "\ninput: 'ATGCCCGCTTT'"
    print "expected output:AAAGCGGGCAT"
    print "actual output:  "+get_reverse_complement('ATGCCCGCTTT')
    print "\ninput: 'TCTACT'"
    print "expected output:AGTAGA"
    print "actual output:  "+get_reverse_complement('TCTACT')  

def rest_of_ORF(dna):
    """ Takes a DNA sequence that is assumed to begin with a start codon and returns
        the sequence up to but not including the first in frame stop codon.  If there
        is no in frame stop codon, returns the whole string.
        
        dna: a DNA sequence
        returns: the open reading frame represented as a string   
    """
    #Stop codons:TAG, TAA, or TGA 
    stopcodons = ['TAG','TAA','TGA']
    stopindex = len(dna) # initialize stopindex to len(dna) so that if not found, return whole string
    for i in range((len(dna))/3): #only checks multiples of 3
        if dna[(3*i):(3*i)+3] in stopcodons: 
            stopindex = i*3
            break #stops at first frame   
    return dna[0:stopindex] 

def rest_of_ORF_unit_tests():
    """ Unit tests for the rest_of_ORF function """
    print "input: 'TCCGACTGAC'" 
    print "expected output:TCCGAC"
    print "actual output:  "+rest_of_ORF('TCCGACTGAC')
    print "\ninput: 'AATTACACACATAACACTTGA'"
    print "expected output:AATTACACACA"
    print "actual output:  "+rest_of_ORF('AATTACACACATAACACTTGA')
    print "\ninput: 'AAGTGTTGACCAACCCCACACCAGTAG'"
    print "expected output:AAGTGT"
    print "actual output:  "+rest_of_ORF('AAGTGTTGACCAACCCCACACCAGTAG')
    print "\ninput: 'TCTACT'"
    print "expected output:TCTACT"
    print "actual output:  "+rest_of_ORF('TCTACT')  
        
def find_all_ORFs_oneframe(dna):
    """ Finds all non-nested open reading frames in the given DNA sequence and returns
        them as a list.  This function should only find ORFs that are in the default
        frame of the sequence (i.e. they start on indices that are multiples of 3).
        By non-nested we mean that if an ORF occurs entirely within
        another ORF, it should not be included in the returned list of ORFs.
        
        dna: a DNA sequence
        returns: a list of non-nested ORFs
    """
    i =0
    x =[]
    if type(dna)==None:#for errors that only came up with randomized strings
            return x
    while i<len(dna):#i is the index that will be moved along the strand
    #note that it shouldn't be <= because then the last frame would be empty/the last stop codon
        frame=rest_of_ORF(dna[i:len(dna)]) #takes remaining portion
        toAdd=frame
        startfound=False
        while startfound==False:
            if toAdd[0:3]=="ATG":#If leading codon isn't ATG, remove
                x.append(toAdd)
                startfound=True                
            else:
                toAdd=toAdd[3:len(toAdd)]
            if toAdd=='' or len(toAdd)<=2:                
                startfound=True
        i=i+len(frame)+3#to the stop index        
    return x
     
def find_all_ORFs_oneframe_unit_tests():
    """ Unit tests for the find_all_ORFs_oneframe function """
    print "input: 'ATGCATGAATGTAGATAGATGTGCCC'" 
    print "expected output:\n['ATGCATGAATGTAGA', 'ATGTGCCC']"
    print "actual output:  "
    print find_all_ORFs_oneframe('ATGCATGAATGTAGATAGATGTGCCC')
    
    print "Note: following unit tests not valid after front-truncating for ATG"
    print "\ninput: 'AATCGGTGA'"
    print "expected output:\n['AATCGG']"
    print "actual output:  "
    print find_all_ORFs_oneframe('AATCGGTGA')
    print "\ninput: 'AAGTGTTGACCAACCCCACACCAGTAG'"
    print "expected output:\n['AAGTGT', 'CCAACCCCACACCAG']"
    print "actual output:  "
    print find_all_ORFs_oneframe('AAGTGTTGACCAACCCCACACCAGTAG')
    print "\ninput: 'TCTACTTGAAAA'"
    print "expected output:\n['TCTACT', 'AAA']"
    print "actual output:  "
    print find_all_ORFs_oneframe('TCTACTTGAAAA') 
    
def find_all_ORFs(dna):
    """ Finds all non-nested open reading frames in the given DNA sequence in all 3
        possible frames and returns them as a list.  By non-nested we mean that if an
        ORF occurs entirely within another ORF and they are both in the same frame,
        it should not be included in the returned list of ORFs.
        
        dna: a DNA sequence
        returns: a list of non-nested ORFs
    """
    x=[]
    for i in range(3):
        x=x+find_all_ORFs_oneframe(dna[i:len(dna)]) #concatenates lists but 
        #does not append them because that would create nested lists
    return x 

def find_all_ORFs_unit_tests():
    """ Unit tests for the find_all_ORFs function """
    print "input: 'CCCCCCCCCATGCCCTAGATGTAG'" 
    print "expected output:\n['ATGCCC','ATG']"
    print "actual output:  "
    print find_all_ORFs('CCCCCCCCCATGCCCTAGATGTAG')
    print "\ninput: 'ATGCATGAATGTAG'"
    print "expected output:\n['ATGCATGAATGTAG', 'ATGAATGTAG', 'ATG']"
    print "actual output:  "    
    print find_all_ORFs('ATGCATGAATGTAG')

def find_all_ORFs_both_strands(dna):
    """ Finds all non-nested open reading frames in the given DNA sequence on both
        strands.
        
        dna: a DNA sequence
        returns: a list of non-nested ORFs
    """
    return find_all_ORFs(dna)+find_all_ORFs(get_reverse_complement(dna))

def find_all_ORFs_both_strands_unit_tests():
    """ Unit tests for the find_all_ORFs_both_strands function """
    print "input: 'ATGGCCTGATCACAT'" 
    print "expected output:\n['ATGGCC', 'ATGGGG']"
    print "actual output:  "
    print find_all_ORFs_both_strands('ATGGCCTGATCACCCCAT')

def longest_ORF(dna):
    """ Finds the longest ORF on both strands of the specified DNA and returns it
        as a string"""
    return max(find_all_ORFs_both_strands(dna))

def longest_ORF_unit_tests():
    """ Unit tests for the longest_ORF function """
    print "input: 'ATGCGAATGTAGCATCAAA'" 
    print "expected output:\n['ATGCTACATTCGCAT']"
    print "actual output:"
    print "  "+longest_ORF('ATGCGAATGTAGCATCAAA')

def longest_ORF_noncoding(dna, num_trials):
    """ Computes the maximum length of the longest ORF over num_trials shuffles
        of the specfied DNA sequence
        
        dna: a DNA sequence
        num_trials: the number of random shuffles
        returns: the maximum length longest ORF """
    maxlength=0
    longestORF = ''
    for i in range(num_trials):
        q=list(dna)
        shuffle(q)
        q=collapse(q)
        v =longest_ORF(q)
        checklength = len(v)
        if checklength>maxlength:
            maxlength = checklength
            longestORF = v
    return [longestORF,maxlength]
    #Using the salmonella sequence and 1500 trials, a maxlength of 558 found

def gene_finder(dna, threshold):
    """ Returns the amino acid sequences coded by all genes that have an ORF
        larger than the specified threshold.
        
        dna: a DNA sequence
        threshold: the minimum length of the ORF for it to be considered a valid
                   gene.
        returns: a list of all amino acid sequences whose ORFs meet the minimum
                 length specified.
    """
    x=[]
    y=[]
    strands=find_all_ORFs_both_strands(dna)
    for i in range(len(strands)):
        if len(strands[i])>threshold:
            x.append(strands[i])
    for i in range(len(x)):
        y.append(coding_strand_to_AA(x[i]))
    return y
    
