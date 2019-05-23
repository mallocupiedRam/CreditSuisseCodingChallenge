from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
import os
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

  
warnings.filterwarnings(action = 'ignore') 
  
import gensim 
from gensim.models import Word2Vec 

#avearges the word vectors of the words in the document
def doc_vec(word2vec_model, doc):
    doc = doc.split()
    #filters out words not in the vocabulary
    doc = [word for word in doc if word in word2vec_model.wv.vocab]
    return np.mean(word2vec_model[doc], axis=0)


#trains a skipgram model based on the text in the files in arg1 and saves it to arg2
def train_model(path_to_files, path_to_save):
    all_files = ""

    #concatenates all files
    for f in os.listdir(path_to_files):
        if(f[0]!="."):
            sample = open(path_to_files+f, "r") 
            text = sample.read()
            all_files +=  text
        all_files+=" "
    
    
    data = [] 
    
    # iterate through each sentence in the file 
    for i in sent_tokenize(f): 
        temp = [] 
        
        # tokenize the sentence into words 
        for j in word_tokenize(i): 
            temp.append(j.lower()) 
    
        data.append(temp) 

    #generates model
    model1 = gensim.models.Word2Vec(data, min_count = 1, size = 50, window = 4,sg=1)
    
    #saves model
    model1.save(path_to_save)



#prints the best resumes for the listing passed in using document similarity
def output_best(path_to_resumes, listing_text):
    all_files = ""
    files = []
    resume_cnt=0
    resumes = []
    

    #opens all resumes
    for f in os.listdir(path_to_resumes):
        if(f[0]!="."):
            sample = open(path_to_resumes+f, "r") 
            text = sample.read()
            all_files +=  text
            files.append(text)
            resume_cnt+=1
            resumes.append((f,text))
        all_files+=" "
    #adds listing to files list
    files.append(listing_text)

    #loads model
    model1 = gensim.models.Word2Vec.load("word2vec.model")

    #calculates avgs of documents
    centroid_vec = []
    for f in files:
        centroid_vec.append(doc_vec(model1,f))
    
    #sim_matr[i][j] is the similarity of ith doc vector and jth doc vector
    sim_matr = cosine_similarity(np.array(centroid_vec))

    #lables the similarity vector for the listing so we can sort the matrix.
    sim_enum = list(enumerate(sim_matr[-1][:resume_cnt]))

    #sorts from high to low by similarity
    sim_enum.sort(key = lambda x: -x[1])
    
    #prints top three resumes
    print([resumes[x[0]][0] for x in sim_enum[:3]])


