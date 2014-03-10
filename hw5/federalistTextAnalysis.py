# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 16:36:20 2014

@author: atproofer - Michael Bocamazo
"""

"""README: This project is an attempt to tackle/recreate the solution to the
 Federalist author identification problem.
 Use the dispute_comparison() function with a pre-defined filter list to see
 results of project"""


"""Here, specific text snippets and phrases relevant to parsing the file:
URL: 'http://www.gutenberg.org/files/1404/1404-h/1404-h.htm'
Start of first full text:
"For the Independent Journal. Saturday, October 27, 1787"
(need to skip past contents section, fine to break off the title of the first text)
Separator of Each paper:
    "FEDERALIST No. %s" %(paper number)
End of last text:
    "End of the Project Gutenberg EBook" 
    
    Data Structure plan:
        Import full text file from gutenberg
        parse file into string block
        loop through string, separate out papers into discrete strings
        store as list of individual papers
        create bag of words for each paper
        store those dictionaries as elements in a list
        create bag of words for entire series of papers
        create new list of these bags
            to compare each normalized frequency with the full set
            
        filter for function words to control for content - need to find 
        
        distance function: could be linear, rectilinear, squared, cosine similarity
        possibly make distance map        
        
    Other ideas for style analysis: ("Writer Invariants")
        sentence length
        frequency of non-period punctuation
        number of different words normalized (~nec linearly) for length of document
        stdev of frequencies (?)
        area under the curve v triangular area (like Gini coefficient) for frequencies
        
        'Bailey (1979) lists the general properties for such variables: “They should be
        salient, structural, frequent, and relatively immune from conscious control”.'
        http://www.stat-d.si/mz/mz4.1/dabagh.pdf
        I would add: high consistency within author, high variability among authors as primary selection criteria
        Distribution of Word Length commonly used - how can this be something other
        than lossy-compressed tf-idf?... Or do authors' vocabularies actually favor a word length?
            
    Further explorations: 
        Can possibility of collaboration be judged?
        Conjecture: separate disputed texts into paragraphs/super-paragraph sections and run
        analysis on those... Could take permutations of paragraph breaks and analyze each
        If multiple authors wrote separate paragraphs, this could work
        
        -Do multi-dimensional map of texts
        
    Project Notes:
        Should probably make a separate text initialize function rather than
        initializing all of it multiple times
        
        Probable output at this point: 16x3 matrix comparing disputed docs
        with average of knowns
        
        MDS doesn't work well because a few documents are far off, the rest
        clustered tightly, also currently mostly a black box, and don't yet know
        how to label points individually
        """

"""
SS: Mike, this is an outstanding job!! Well done. This is an extremely compelling project and your 
    exectuion of this project by yourself in this limited amount of time is amazing. 

    Your code is well structured, commented, and easy to follow. And it functionally is sounds. 

    Great, great job :) 
"""
        
from pattern.web import *
from pattern.en import *
import string
import os
import pickle
import random
import numpy as np

## SS: Because you imported numpy as np, I needed to change your zeros() calls to np.zeros() to have 
##     it working from the terminal

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

standard_filters = ['EnglishDeterminers.txt','EnglishConjunctions.txt']
all_filters = ['EnglishPronouns.txt','EnglishAuxiliaryVerbs.txt','EnglishPrepositions.txt','EnglishQuantifiers.txt','EnglishDeterminers.txt','EnglishConjunctions.txt']
nonstandard_filters = ['EnglishPronouns.txt','EnglishAuxiliaryVerbs.txt','EnglishPrepositions.txt','EnglishQuantifiers.txt']
own_filters = ['OwnFilters.txt']


disputed_papers=([18,19,20]+range(49,59)+[62,63,64])
Hamilton = range(6,10)+range(21,37)+range(65,86)+[1, 11, 12, 13, 15, 16, 17, 59, 60, 61]
Madison = ([10,14]+range(37,50))
Jay=[2,3,4,5]#Some sources give 64 as definitive for Jay
scholar = {64:'J',49:'M',50:'M',51:'M',52:'M',53:'M',54:'M',55:'M',56:'M',57:'M',58:'M',62:'M',63:'M',18:'H/M',19:'H/M',20:'H/M'}
 

def pull_text(site,filename):
    """Pulls the text from the URL (string) and stores in filename, need
    filetype specified in filename"""
    x=URL(site).download()
    f = open(filename,'w')
    f.write(x)
    f.close()
#    Check:
#    g = open("FederalistText.txt",'r')
#    fulltext = g.read()
#    g.close()
#    print fulltext

def iso_text(site,filename,boilerstart,boilerend):
    """From a site and w/ filename, isolates and parses into body+bag of words.
    Returns body, histogram of full text."""
    if not(os.path.exists(filename)):
        pull_text(site,filename)
    f = open(filename,'r')
    fulltext = f.read()
    f.close()
    bodystart=fulltext.find(boilerstart)
    bodyend=fulltext.find(boilerend)
    body=fulltext[bodystart:bodyend]
    [bag_of_words,hist]=parse_text(body)
    return [body,bag_of_words,hist]
    
def parse_text(input_text):
    #This section adapted from Allen Downey's "Think Python" Ch. 13
    #Sanitizing doesn't remove some tags, really only works in body of text
    hist={} #initialize histogram
    input_text = input_text.replace('-','') #separate dashed words into whitespace
    input_text = input_text.replace('&mdash',' ') #Added check for &mdash, replacing — didn't work
    bag_of_words = input_text.split()
    for word in bag_of_words: #loop through a list of the words in the body
        word = word.strip(string.punctuation + string.whitespace) #sanitize
        word = word.lower() #regularize to lower case for single counting
        if not('=' in word): #Added check to remove tags
            hist[word] = hist.get(word, 0) + 1 #add word to current key, initialize +1 otherwise
    return [bag_of_words,hist]
    
def letters_separate(body):
    """Takes in body text, returns lists of documents and of corresponding
    bags and histograms"""
    #85 letters
    start_index=0
    doc_list=[]
    bag_list=[]
    hist_list=[]
    for i in range(1,85):
        next_index = body.find("FEDERALIST No. %d"%(i+1))#stop at the next doc
        doc_list.append(body[start_index:next_index])
        [doc_bag,doc_hist] = parse_text(doc_list[i-1]) #index will be one less than doc number
        bag_list.append(doc_bag)
        hist_list.append(doc_hist)
        start_index=next_index
    #85th doc case
    doc_list.append(body[start_index:len(body)-1])
    [doc_bag,doc_hist] = parse_text(doc_list[84])#This is where the 85th should end up
    bag_list.append(doc_bag)
    hist_list.append(doc_hist)
    return [doc_list, bag_list, hist_list]
    """Checks:
        for j in range(len(doc_list)):
            sum += len(doc_list[j])
        compare sum to 
        len(body)
        result: 1289251 v 1289252 ... good enough
        
        for g in range(len(bag_list)):
            j+=len(b[g])
        v len(body.split())
        197335 v 197260 ... difference should be from sanitation 
        
        sum(hist_list[i].values()) v len(bag_list[i])
        ... consistent difference of 3, maybe 3 tags in each document?
        so should probably use hist because it's sanitized
        """
        
def parse_lang_files(filename):
    """For parsing the function word lists.  Input the list filename. 
    Returns the word list.
    should use: determiners, conjunctions - least content-dependent.
    Doesn't word for multi-word phrases, which are also not being analyzed"""
    f = open(filename,'r')
    fulltext = f.read()
    f.close()
    words_index = fulltext.rfind('/')+1 #based on the formatting of the machine files
    [bag,hist]=parse_text(fulltext[words_index:len(fulltext)])
    return bag

def get_filter_words(filter_list):
    """Filter list is a list of filenames, returns list of words"""
    filter_words = []
    for filename in filter_list:
        filter_words.extend(parse_lang_files(filename))
    return list(set(filter_words)) #removes duplicates
    
def filter_function_words(hist,filter_words):
    """Takes in a histogram for a sequence and a filter word list
    and retains only keys in the filter list."""
    output_hist = {}
    for word in filter_words: #Want full list, not just matches, to compare two different texts
        output_hist[word]=0
    for key in hist:
        if key in filter_words:
            output_hist[key]=hist[key]            
    return output_hist
    
def calc_normalized_frequency(hist):
    """Takes in a histogram and divides all terms by the number of terms in
    the doc, returns histogram w/ normed values"""
    output_hist = {}
    for key in hist:
        output_hist[key] = hist[key]/float(sum(hist.values()))
    return output_hist
    
def calc_cosine_similarity(hist1,hist2):
    """Takes in two dicts of standard keys and returns their similarity.
    Should use after filter_function_words."""
    cos_sim = 0
    for key in hist1:
        cos_sim+=hist1[key]*hist2[key]
    v1=np.array(hist1.values())
    m1=np.linalg.norm(v1)
    v2=np.array(hist2.values())
    m2=np.linalg.norm(v2)
    cos_sim = cos_sim/(float(m1)*m2)
    return cos_sim
        
def cos_compare_docs(doc1,doc2):
    """Input the number of two documents to compare (ints).  Returns the cosine similarity.
    Takes in the one-indexed numbering."""
    [body,bag_of_words,hist] = iso_text('http://www.gutenberg.org/files/1404/1404-h/1404-h.htm','FederalistText.txt',"For the Independent Journal. Saturday, October 27, 1787","End of the Project Gutenberg EBook")
    [doc_list, bag_list, hist_list]=letters_separate(body)
    filter_words=get_filter_words(standard_filters)#change input here for different word list
    hist1 = filter_function_words(hist_list[doc1-1],filter_words)
    hist2 = filter_function_words(hist_list[doc2-1],filter_words)
    cos_sim = calc_cosine_similarity(hist1,hist2)
    return cos_sim
    #Simple Check: make sure cos of the same docs is 1.0
    
def add_hists(hist1,hist2):
    """Takes in two histograms add sums the values of the matching keys, returns summed hist"""
    out_hist = hist2
    for key in hist1:
        if key not in out_hist:
            out_hist[key] = hist1[key]
        else:
            out_hist[key] += hist1[key]
    return out_hist
    
def add_hists_freq(hist1,hist2,n):
    """Takes in two histograms add sums the values of the matching keys, 
    returns summed hist... used in mean freq with normalization"""
    out_hist = hist2
    for key in hist1:
        if key not in out_hist:
            out_hist[key] = hist1[key]/float(n)
        else:
            out_hist[key] += hist1[key]/float(n)
    return out_hist
    

## SS: Since you're duplicating the first few lines of calc_does_mean() in calc_does_mean2()
##     it might have been a good idea to merge the two into one function and then have another input
##     that would specify which one you want to execute
    
def calc_docs_mean(authored_list,filter_list):
    """Input a list of document numbers (one indexed) and filter_words,
    returns a dict of the average frequencies of the filtered words.
    That is, reconstructs the hist of doc set, and normalizes it."""
    [body,bag_of_words,hist] = iso_text('http://www.gutenberg.org/files/1404/1404-h/1404-h.htm','FederalistText.txt',"For the Independent Journal. Saturday, October 27, 1787","End of the Project Gutenberg EBook")
    [doc_list, bag_list, hist_list]=letters_separate(body)
    filter_words=get_filter_words(filter_list)
    n=len(doc_list)
    output_hist = {}    
    for doc in authored_list:
        z=calc_normalized_frequency(hist_list[doc-1])
        filtered_hist = filter_function_words(z,filter_words)
        #The normalized, filtered hist of the doc
        output_hist=add_hists_freq(output_hist,filtered_hist,n)
    return output_hist
    """Note: Averages the frequencies among docs, rather than finding frequencies
    in the summed doc.  So weights shorter docs slightly more heavily.  Could be fixed,
    but not really important"""
    
def calc_docs_mean2(authored_list,filter_list):
    """Input a list of document numbers (one indexed) and filter_words,
    returns a dict of the average frequencies of the filtered words.
    That is, reconstructs the hist of doc set, and normalizes it."""
    [body,bag_of_words,hist] = iso_text('http://www.gutenberg.org/files/1404/1404-h/1404-h.htm','FederalistText.txt',"For the Independent Journal. Saturday, October 27, 1787","End of the Project Gutenberg EBook")
    [doc_list, bag_list, hist_list]=letters_separate(body)
    filter_words=get_filter_words(filter_list)
    
    full_hist = {}    
    for doc in authored_list:        
        full_hist=add_hists(full_hist,hist_list[doc-1])
    sum_words = sum(hist.values())
    
    filtered_hist = filter_function_words(full_hist,filter_words)
    
    output_hist = {}
    for key in filtered_hist:
        output_hist[key] = filtered_hist[key]/float(sum_words)

    return output_hist

        
def generate_diff_matrix():
    """Loops through all the documents and returns a matrix of cosine similarities
    between them, returns diff_matrix"""
    diff_matrix = np.zeros((85,85))    
    for i in range(0,85):
        for j in range(0,85):
            diff_matrix[i,j]=cos_compare_docs(i+1,j+1)
    return diff_matrix
    
def fast_generate_diff_matrix():
    """Saves looping time by opening the file only once"""
    [body,bag_of_words,hist] = iso_text('http://www.gutenberg.org/files/1404/1404-h/1404-h.htm','FederalistText.txt',"For the Independent Journal. Saturday, October 27, 1787","End of the Project Gutenberg EBook")
    [doc_list, bag_list, hist_list]=letters_separate(body)
    filter_words=get_filter_words(nonstandard_filters)
    diff_matrix = np.zeros((85,85))    
    for i in range(0,85):
        for j in range(0,85):
            hist1 = filter_function_words(hist_list[i],filter_words)
            hist2 = filter_function_words(hist_list[j],filter_words)
            diff_matrix[i,j]=calc_cosine_similarity(hist1,hist2)
    return diff_matrix

def dispute_comparison(filter_list):
    """Displays results of comparison.  Order is paper, Jay, Hamilton, Madison score,
    highest score.  Experiment with filter list to see different approaches.
    Refer to the scholar dict for comparison with accepted results."""
    [body,bag_of_words,hist] = iso_text('http://www.gutenberg.org/files/1404/1404-h/1404-h.htm','FederalistText.txt',"For the Independent Journal. Saturday, October 27, 1787","End of the Project Gutenberg EBook")
    [doc_list, bag_list, hist_list]=letters_separate(body)
    filter_words=get_filter_words(filter_list)
    mean_Jay = calc_docs_mean2(Jay,filter_list)
    mean_Hamilton = calc_docs_mean2(Hamilton,filter_list)
    mean_Madison = calc_docs_mean2(Madison,filter_list)
    known_vector = [mean_Jay,mean_Hamilton,mean_Madison]
    results_matrix = np.zeros((len(disputed_papers),5))
    for k in range(len(disputed_papers)):
        results_matrix[k,0]=disputed_papers[k]
        
#    Need to use string labels in presentation
#        results_vector[k,1]=scholar[disputed_papers[k]]
#    results_matrix[0,0]='Doc #'
#    results_matrix[0,1]='Scholar'
#    results_matrix[0,2]='Jay'
#    results_matrix[0,3]='Hamilton'
#    results_matrix[0,4]='Madison'
    for i in range(len(disputed_papers)):        
        paper_hist = filter_function_words(hist_list[disputed_papers[i]-1],filter_words)
        y=0
        for j in range(len(known_vector)):
            z = calc_cosine_similarity(paper_hist,known_vector[j])
            results_matrix[i,j+1] = z
            if z>y:
                y=z
                x = j+1
        results_matrix[i,4]=x #allows seeing the highest match quickly
            
    return results_matrix
            
    
    
def plot_MDS():
    """Plots the difference matrix with Multi-Dimensional Scaling"""
    diff_matrix = fast_generate_diff_matrix()
    X_true = diff_matrix
    similarities = euclidean_distances(diff_matrix)
    seed = 1
    

    mds = manifold.MDS(n_components=1, max_iter=3000, eps=1e-9, random_state=seed,
                       dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(similarities).embedding_
    
#    nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
#                        dissimilarity="precomputed", random_state=2, n_jobs=1,
#                        n_init=1)
#    npos = nmds.fit_transform(similarities, init=pos)
    
    # Rescale the data
    pos *= np.sqrt((X_true ** 2).sum()) / np.sqrt((pos ** 2).sum())
#    npos *= np.sqrt((X_true ** 2).sum()) / np.sqrt((npos ** 2).sum())
    
    # Rotate the data
    clf = PCA(n_components=2)
    X_true = clf.fit_transform(X_true)
    
    pos = clf.fit_transform(pos)
#    
#    npos = clf.fit_transform(npos)
    
    fig = plt.figure(1)
    ax = plt.axes([0., 0., 1., 1.])
    
    plt.scatter(X_true[:, 0], X_true[:, 1], c='r', s=20)
#    plt.scatter(pos[:, 0], pos[:, 1], s=20, c='g')
#    plt.scatter(npos[:, 0], npos[:, 1], s=20, c='b')
    plt.legend(('True position'), loc='best')
    
    similarities = similarities.max() / similarities * 100
    similarities[np.isinf(similarities)] = 0
    
    # Plot the edges
    start_idx, end_idx = np.where(pos)
    #a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [[X_true[i, :], X_true[j, :]]
                for i in range(len(pos)) for j in range(len(pos))]
    values = np.abs(similarities)
    lc = LineCollection(segments,
                        zorder=0, cmap=plt.cm.hot_r,
                        norm=plt.Normalize(0, values.max()))
    lc.set_array(similarities.flatten())
    lc.set_linewidths(0.5 * np.ones(len(segments)))
    ax.add_collection(lc)
    
    plt.show()