#import module_manager
#module_manager.review()
import PyPDF2
import textract
import sys
import os
import docx2txt
import json
import random
import logging
import re
import spacy
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.metrics import accuracy_score

from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords

global_text = ""
global_file = ""
nlp1 = spacy.load("Entity-Recognition-In-Resumes-SpaCy-master/my_model")
if 'ner' not in nlp1.pipe_names:
    ner = nlp1.create_pipe('ner')
    nlp1.add_pipe(ner, last=True)
       

def createParsed(filename):
    final = ''
    file = ""
    
    parsed = filename.split(".")
    extension = parsed[len(parsed) -1]
    parsed = parsed[:-1]
    s = "."
    file = s.join(parsed)
    global_file = file
    print(extension)
    
    if (extension == "docx" or extension == "doc"):
        text = docx2txt.process(filename)
        final = ""
        for char in text:
            if (32 <= ord(char) <= 126 or ord(char) == 10):
                final += char
        
        
    elif (extension == 'pdf'):
        pdfFileObj = open(filename, 'rb');
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj);
        num_pages = pdfReader.numPages
        count = 0
        text = ""
        
        while count < num_pages:
            pageObj = pdfReader.getPage(count)
            count+=1
            text+= pageObj.extractText()
        
        if text != "":
            text = text
        
        else: 
            text = textract.process(fileurl, method = 'tesseract', language = 'eng')
        
        print (1)
        for char in text:
            if (32 <= ord(char) <= 126):
                final += char
    global_text = final

    f=open(global_file+".txt","w", encoding="utf-8")
    doc_to_test=nlp1(global_text)
    d={}
    for ent in doc_to_test.ents:
        d[ent.label_]=[]
    for ent in doc_to_test.ents:
        d[ent.label_].append(ent.text)
    
    for i in set(d.keys()):
    
        f.write("\n\n")
        f.write(i +":"+"\n")
        for j in set(d[i]):
            f.write(j.replace('\n','')+"\n")
    print (d)

def generateText():
    for filename in os.listdir("Resumes"):
        if (filename != ".DS_STORE"):
            createText("Resumes/" + filename)
            
            