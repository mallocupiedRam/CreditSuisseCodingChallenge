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
#load the trained models
nlp1 = spacy.load("Entity-Recognition-In-Resumes-SpaCy-master/my_model1")
nlp2 = spacy.load("Entity-Recognition-In-Resumes-SpaCy-master/my_model")
if 'ner' not in nlp1.pipe_names:
    ner = nlp1.create_pipe('ner')
    nlp1.add_pipe(ner, last=True)
       

def createParsed(filename):
    final = ''
    file = ""
    #find the extension
    parsed = filename.split(".")
    extension = parsed[len(parsed) -1]
    parsed = parsed[:-1]
    s = "."
    file = s.join(parsed)
    global_file = file
    
    #parse the file into strings depending on the file type
    if (extension == "docx" or extension == "doc"):
        text = docx2txt.process(filename)
        final = ""
        #omit non-ascii characters
        for char in text:
            if (32 <= ord(char) <= 126 or ord(char) == 10):
                final += char
        
    #file type is pdf
    elif (extension == 'pdf'):
        pdfFileObj = open(filename, 'rb');
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj);
        num_pages = pdfReader.numPages
        count = 0
        text = ""
        
        #get text from pdf
        while count < num_pages:
            pageObj = pdfReader.getPage(count)
            count+=1
            text+= pageObj.extractText()
        #detect if text was parseable
        if text != "":
            text = text
        
        else: 
            #use ocr if cannot parse pdf
            text = textract.process(fileurl, method = 'tesseract', language = 'eng')
        #omit non-ascii characters
        for char in text:
            if (32 <= ord(char) <= 126):
                final += char
    global_text = final
    
    #load the two models and run on the string text parsed from files
    doc_to_test=nlp1(global_text)
    doc_to_test2 = nlp2(global_text)
    d={}
    add = True
    #creates a dictionary from the model labels to the data from both models
    for ent in doc_to_test.ents:
        d[ent.label_]=[] #adds the relevant labels from the model
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

            
            