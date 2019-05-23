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
  
all_files = ""
files = []
resume_cnt=0
resumes = []
listings = []

for f in os.listdir("Resumes+listings"):
    if(len(f.split("."))==1 and f!="Listings"):
        sample = open("Resumes+listings/"+f, "r") 
        text = sample.read()
        all_files +=  text
        files.append(text)
        resume_cnt+=1
        resumes.append((f,text))
    all_files+=" "
for f in os.listdir("Resumes+listings/Listings"):
    if(len(f.split("."))==1):
        sample = open("Resumes+listings/Listings/"+f, "r") 
        text = sample.read()
        all_files +=  text
        files.append(text)
        listings.append((f,text))
    all_files+=" "
 
# Replaces escape character with space 
all_files = all_files.lower()
f = all_files.replace("\n", " ") 

most_com = ["the","be","to","of","and","a","in","that","have","i","it","for","not","on","with","as","at","this","but","by"]
for word in most_com:
    f = f.replace(word,"")



data = [] 
  
# iterate through each sentence in the file 
for i in sent_tokenize(f): 
    temp = [] 
      
    # tokenize the sentence into words 
    for j in word_tokenize(i): 
        temp.append(j.lower()) 
  
    data.append(temp) 
  
# Create CBOW model 
model1 = gensim.models.Word2Vec(data, min_count = 1,  
                              size = 50, window = 4,sg=1)
#model1 = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)


def doc_vec(word2vec_model, doc):
    doc = doc.split()
    doc = [word for word in doc if word in word2vec_model.wv.vocab]
    return np.mean(word2vec_model[doc], axis=0)

centroid_vec = []
for f in files:
    centroid_vec.append(doc_vec(model1,f))
#print(centroid_vec)

sim_matr = cosine_similarity(np.array(centroid_vec))


for i in range(resume_cnt,len(files)):
    sim_enum = list(enumerate(sim_matr[i][:resume_cnt]))
    sim_enum.sort(key = lambda x: -x[1])
    #print(sim_enum)
    print(listings[i-resume_cnt][0],":",[resumes[x[0]][0] for x in sim_enum[:3]])

for i in range(resume_cnt):
    sim_enum = list(enumerate(sim_matr[i][resume_cnt:]))
    sim_enum.sort(key = lambda x: -x[1])
    #print(sim_enum)
    print(resumes[i][0],":",[listings[x[0]][0] for x in sim_enum[:2]])

