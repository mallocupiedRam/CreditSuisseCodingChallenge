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
from fuzzywuzzy import fuzz


global_text = ""
global_file = ""
nlp1 = spacy.load("Entity-Recognition-In-Resumes-SpaCy-master/my_model1")
nlp2 = spacy.load("Entity-Recognition-In-Resumes-SpaCy-master/my_model")
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
        
        for char in text:
            if (32 <= ord(char) <= 126):
                final += char
    global_text = final

    doc_to_test=nlp1(global_text)
    doc_to_test2 = nlp2(global_text)
    d={}
    add = True
    for ent in doc_to_test.ents:
        d[ent.label_]=[]
    for ent in doc_to_test.ents:
        d[ent.label_].append(ent.text)
    for ent in doc_to_test2.ents:
        if ent.label_ not in d.keys():
            d[ent.label_] = []
    for ent in doc_to_test2.ents:
        for en in d[ent.label_]:
            if (fuzz.partial_ratio(ent.text, en) > 90):
                add = False
        if (add or d[ent.label_] == []): d[ent.label_].append(ent.text)
    
    return (d)

            
            