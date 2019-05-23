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
import nltk
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.metrics import accuracy_score

import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
# 
# nltk.download( )

#test dictionaries
dict1 = {'Name': ['Dorthy Urda'], 'Years of Experience': ['9 Years'], 'Location': ['Pune'], 'Companies worked at': ['dd', 'dd', 'FI', 'Linux', 'Control'], 'Designation': ['Senior Associate', 'Configuration Manager', 'Configuration Manager', 'Developer, Team Lead', 'Developer, Team Lead', 'FileNet Administrator', 'SAKS Panagon Access'], 'Skills': ['Database', 'Shell script, Perl, SQL  PL/SQL', 'Release and Change Management, ITAO', 'SVN', 'CVS'], 'Degree': ['Bachelor of Engineering (Electronics and Telecommunication), from University of Pune', 'Developer', 'Developer', 'Developer', 'Developer'], 'Graduation Year': ['2.6.32'], 'College Name': ['Version-One', 'UNIX Shell Scripting']}

dict2 = {'Name': ['Beverly Jenkins'], 'Designation': ['Cognizant Technology Solutions', 'Cognizant Technology Solution'], 'Location': ['Pune', 'Syslog'], 'Skills': ['Skill Set\n\n: MS SQL, Windows Server, IIS, UNIX.\n\nTools\n\n: Splunk Enterprise, NIMSOFT, Geneos, CA AUTOSYS, Idash, SOAP UI, Service NOW, IBM WAS console, Aqua Data, Sql Management Studio'], 'Graduation Year': ['2992960316'], 'Degree': ['A N College'], 'College Name': ['Magadh University']}

dict3 = {'Name': ['Johnny Fontenot'], 'Designation': ['Project Manager/Senior Business Analyst in Intellimatch,', 'Associate', 'Senior Consultant', 'Software Engineer', 'Application Developer'], 'Companies worked at': ['Oracle', 'Autosys', 'TCS', 'TCS', 'CDAC', 'Shell Script', 'Shell and', 'Developer', 'Autosys', 'Developer'], 'Location': ['Barcap', 'Pune'], 'Degree': ['Bachelor of Engineering (CSE)', 'Scheduler-'], 'College Name': ['Jiwaji University'], 'Skills': ['Application', 'AVP', 'OMD'], 'Graduation Year': ['2005/08']}

dict4 = {'Name': ['Ignacio Hinson', 'Oracle SQL'], 'Designation': ['Project Manger', 'Project Manager', 'Team Size', 'Project Manager', 'Project Manager', 'Project Manager', 'Asst. Consultant (ASC)- Role- Project Manager', 'Sr. Software Engineer (Role -Associate Project Manager)', 'Technical Support Executive'], 'Companies worked at': ['Oracle', '(Role-', 'DBA', 'Oracle', 'DBA', 'Autosys', 'Oracle', 'Oracle', 'Infosys Ltd', 'Infosys Ltd', 'Infosys Ltd', 'Oracle', 'Infosys'], 'Degree': ['M.C.A from BEHRAMPUR UNIVERSITY, ORISSA with 79% aggregate.', 'B.Sc Hons. (Computer Application)from', 'Associate'], 'Location': ['Pune', 'Pune'], 'Skills': ['Clarify', 'Orange', 'Problem solving and decision making'], 'College Name': ['Attended Quality Calls']}


from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from fuzzywuzzy import fuzz

# Assumes that job descriptions are all in .doc/.docx format
#returns a string containing the text from filename
def parseJob(filename):
    final = ''
    
    parsed = filename.split(".")
    extension = parsed[len(parsed) -1]

    if (extension == "docx" or extension == "doc"):
        text = docx2txt.process(filename)
        #ensures all characters are ascii
        for char in text:
            if (32 <= ord(char) <= 126 or ord(char) == 10):
                final += char
    return final

#runs a string file through a generic NLP model to get the important nouns
def getNouns(raw_parsed_result):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(raw_parsed_result)
    nouns = [chunk.text for chunk in doc.noun_chunks]
    return nouns
    
#generates a score based on number of hits of skills in the parsed resume
#in the important nouns of the job description
def mapSkillstoJob(resume_dict, descrip_nouns):
    #error checking
    if ("Skills" not in resume_dict.keys()):
        return 0
    skills = resume_dict["Skills"]

    if (skills == []): 
        return 0
    score = 0
    for skill in skills:
        for noun in descrip_nouns:
            indiv_skills = skill.split()
            if (skill in noun or noun in skill):
                score += 10 #adds score, scores evenly weighted
                continue
            elif (fuzz.partial_ratio(skill, noun) >= 60):
                #partial score if the match is relatively close
                score += (fuzz.partial_ratio(skill, noun) // 10)
                continue
            else: 
                for indiv_skill in indiv_skills:
                    if (indiv_skill in noun or noun in indiv_skill):
                        score += 10
                    elif (fuzz.partial_ratio(indiv_skill, noun) >= 60):
                        #same as above
                        score += (fuzz.partial_ratio(indiv_skill, noun) // 10)
    return score
                        
#Computes total sum in an array
def aggregate(A):
    sum = 0
    for i in range(len(A)):
        if i == 0 or i == 4 or i == 8: 
            sum+=2*A[i] 
        else: sum+=A[i]
    return sum

#does a fuzzy string compare 
def fcompare(u,v):
    return fuzz.partial_ratio(u,v)

#calculates the match score in the designation category in the resume
def designationScore(resume_dict, descrip_nouns):
    #error checking
    if ("Designation" not in resume_dict.keys()):
        return 0
    designation = resume_dict["Designation"]

    if (designation == []): 
        return 0
    designationList = designation
    nounsList = descrip_nouns
    
    #sorts the designation list and noun list
    SDL = sorted(designationList)
    SNL = sorted(nounsList)
    A = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    sum = 0
    max = 0
    for i in range(len(SDL)):
        for j in range(len(SNL)):
            for k in range(len(A)):
                a = i-1+(k%3)
                b = j-1+(k//3)
                if (0<=a and a<len(SDL) and 0<=b and b<len(SNL)):
                    q = fcompare(SDL[a], SNL[b])
                    if q<30: A[k] = 0
                    else: A[k] = q
                else: A[k] = 0

            sum+=aggregate(A)
        if sum>max: max = sum
        sum = 0
    return max 

#computes a score if prev job category parsed by resume is a big company
def getPrevJobScore(resume_dict):
    #subset of big companies that will give points
    bigCompanies = ['Apple', 'Google', 'Microsoft', 'Tencent', 'Facebook', 'Samsung',
                    'Intel', 'Taiwan Semiconductor Manufacturing', 'Cisco', 
                    'Oracle', 'IBM', 'NVIDIA', 'SAP', 'Adobe', 'Texas Instruments',
                    'Broadcom', 'Accenture', 'Salesforce', 'Qualcomm', 'ASML',
                    'Sony', 'Nintendo', 'HP', 'Hewlett Packard', 'Amazon', 'Dropbox',
                    'Yelp', 'eBay', 'Twitter', 'AirBnB', 'Spotify', 'Snapchat', 
                    'Quora', 'Lyft', 'Uber', 'Netflix', 'General Electric', 
                    'Deloitte', 'P&G', 'Tata', 'General Electric', 'McKinsey',
                    'Goldman Sachs', 'Credit Suisse', 'Deutsche Bank', 'JP Morgan',
                    'Morgan Stanley', 'Barclays', 'Bank of America', 'Citi', 
                    'UBS', 'Merill Lynch', 'Boston Dynamics'
                    ]
    #error checking
    if ("Companies worked at" not in resume_dict.keys()):
        return 0
    companies = resume_dict["Companies worked at"]

    if (companies == []):
        return 0
        
    score = 0
    for company in companies:
        total = 0
        for word in company.split():
            if (word in bigCompanies):
                total = 10
        score += total
    #receives points for each company worked for
    return score // 10

#generates score based on matches in major in parsed resume to parsed job description

def getMajorScore(resume_dict, descrip_nouns):
    if ("Degree" not in resume_dict.keys()):
        return 0
    degree = resume_dict["Degree"]

    if (degree == []): 
        return 0
    score = 0
    for deg in degree:
        for noun in descrip_nouns:
            deg_part = deg.split()
            if (deg in noun or noun in deg):
                score += 10
            elif (fuzz.partial_ratio(deg, noun) >= 60):
                #partial score for a fuzzy string equal
                score += (fuzz.partial_ratio(deg, noun) // 10)
            else: 
                for d in deg_part:
                    if (d in noun or noun in d):
                        score += 10
                    elif (fuzz.partial_ratio(d, noun) >= 60):
                        score += (fuzz.partial_ratio(d, noun) // 10)
    return score

#sums the total score for each labels in the parsed resume to get an aggregate score
#of match based on the resume and the job
def getAggregateScore(resume_dict, filename):
    descrip_string = parseJob(filename)
    descrip_nouns = getNouns(descrip_string)
    score = 0
    score = (score + mapSkillstoJob(resume_dict, descrip_nouns) + 
             designationScore(resume_dict, descrip_nouns) + 
             getPrevJobScore(resume_dict) + 
             getMajorScore(resume_dict, descrip_nouns)
             )
    return score

#runs through all the job descriptions for a specific resume, and returns the 
#3 jobs that have the highest match value
def getTopJobs(resume_dict):
    maxScore = [0, 0, 0]
    matchFile = ["", "", ""]
    for filename in os.listdir("Job Descriptions"):
        if filename != ".DS_STORE":
            score = getAggregateScore(resume_dict, "Job Descriptions/"
                    + filename)
            
            if 0 in maxScore:
                ind = maxScore.index(0)
                maxScore[ind] = score
                matchFile[ind] = filename
            else:    
                for ind in range(len(maxScore)):
                    if (score > maxScore[ind]):
                        maxScore[ind] = score
                        matchFile[ind] = filename
                        break
                    
    return matchFile

     