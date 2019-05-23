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

dict1 = {'Name': ['Dorthy Urda'], 'Years of Experience': ['9 Years'], 'Location': ['Pune'], 'Companies worked at': ['dd', 'dd', 'FI', 'Linux', 'Control'], 'Designation': ['Senior Associate', 'Configuration Manager', 'Configuration Manager', 'Developer, Team Lead', 'Developer, Team Lead', 'FileNet Administrator', 'SAKS Panagon Access'], 'Skills': ['Database', 'Shell script, Perl, SQL  PL/SQL', 'Release and Change Management, ITAO', 'SVN', 'CVS'], 'Degree': ['Bachelor of Engineering (Electronics and Telecommunication), from University of Pune', 'Developer', 'Developer', 'Developer', 'Developer'], 'Graduation Year': ['2.6.32'], 'College Name': ['Version-One', 'UNIX Shell Scripting']}

dict2 = {'Name': ['Beverly Jenkins'], 'Designation': ['Cognizant Technology Solutions', 'Cognizant Technology Solution'], 'Location': ['Pune', 'Syslog'], 'Skills': ['Skill Set\n\n: MS SQL, Windows Server, IIS, UNIX.\n\nTools\n\n: Splunk Enterprise, NIMSOFT, Geneos, CA AUTOSYS, Idash, SOAP UI, Service NOW, IBM WAS console, Aqua Data, Sql Management Studio'], 'Graduation Year': ['2992960316'], 'Degree': ['A N College'], 'College Name': ['Magadh University']}

dict3 = {'Name': ['Johnny Fontenot'], 'Designation': ['Project Manager/Senior Business Analyst in Intellimatch,', 'Associate', 'Senior Consultant', 'Software Engineer', 'Application Developer'], 'Companies worked at': ['Oracle', 'Autosys', 'TCS', 'TCS', 'CDAC', 'Shell Script', 'Shell and', 'Developer', 'Autosys', 'Developer'], 'Location': ['Barcap', 'Pune'], 'Degree': ['Bachelor of Engineering (CSE)', 'Scheduler-'], 'College Name': ['Jiwaji University'], 'Skills': ['Application', 'AVP', 'OMD'], 'Graduation Year': ['2005/08']}

dict4 = {'Name': ['Ignacio Hinson', 'Oracle SQL'], 'Designation': ['Project Manger', 'Project Manager', 'Team Size', 'Project Manager', 'Project Manager', 'Project Manager', 'Asst. Consultant (ASC)- Role- Project Manager', 'Sr. Software Engineer (Role -Associate Project Manager)', 'Technical Support Executive'], 'Companies worked at': ['Oracle', '(Role-', 'DBA', 'Oracle', 'DBA', 'Autosys', 'Oracle', 'Oracle', 'Infosys Ltd', 'Infosys Ltd', 'Infosys Ltd', 'Oracle', 'Infosys'], 'Degree': ['M.C.A from BEHRAMPUR UNIVERSITY, ORISSA with 79% aggregate.', 'B.Sc Hons. (Computer Application)from', 'Associate'], 'Location': ['Pune', 'Pune'], 'Skills': ['Clarify', 'Orange', 'Problem solving and decision making'], 'College Name': ['Attended Quality Calls']}


from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from fuzzywuzzy import fuzz

# Assumes that job descriptions are all in .doc/.docx format
def parseJob(filename):
    final = ''
    
    parsed = filename.split(".")
    extension = parsed[len(parsed) -1]

    if (extension == "docx" or extension == "doc"):
        text = docx2txt.process(filename)
        for char in text:
            if (32 <= ord(char) <= 126 or ord(char) == 10):
                final += char
    return final
    
def getNouns(raw_parsed_result):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(raw_parsed_result)
    nouns = [chunk.text for chunk in doc.noun_chunks]
    return nouns
    

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
                score += 10
                continue
            elif (fuzz.partial_ratio(skill, noun) >= 60):
                score += (fuzz.partial_ratio(skill, noun) // 10)
                continue
            else: 
                for indiv_skill in indiv_skills:
                    if (indiv_skill in noun or noun in indiv_skill):
                        score += 10
                    elif (fuzz.partial_ratio(indiv_skill, noun) >= 60):
                        score += (fuzz.partial_ratio(indiv_skill, noun) // 10)
    return score
                        

def aggregate(A):
    sum = 0
    for i in range(len(A)):
        if i == 0 or i == 4 or i == 8: 
            sum+=2*A[i] 
        else: sum+=A[i]
    return sum

def fcompare(u,v):
    return fuzz.partial_ratio(u,v)

def designationScore(resume_dict, descrip_nouns):
    #error checking
    if ("Designation" not in resume_dict.keys()):
        return 0
    designation = resume_dict["Designation"]

    if (designation == []): 
        return 0
    designationList = designation
    nounsList = descrip_nouns

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


def getPrevJobScore(resume_dict):
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
    return score // 10

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
                score += (fuzz.partial_ratio(deg, noun) // 10)
            else: 
                for d in deg_part:
                    if (d in noun or noun in d):
                        score += 10
                    elif (fuzz.partial_ratio(d, noun) >= 60):
                        score += (fuzz.partial_ratio(d, noun) // 10)
    return score

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

text = "This role is within the Cognitive and Digital Services (CDS) organization which offers an end-to-end service in the development and implementation of Amelia (AI cognitive agent), Contact Center Modernization, data science, machine learning and advanced analytics to the firm. This role reports directly to the Head of the Cognitive and Digital Services organization while also aligning to the organizations Strategy and Architecture discipline for technology continuity. You will be a key individual within our Strategy and Architecture team that has responsibility for the technology strategy, roadmap, feature set, along with the service pricing and TCO.  The role will entail: • Support the Service by developing the strategy and architecture for the service technology and aligned capabilities. • Actively engage with Credit Suisse Customers in partnership with Customer Relationship Management to ensure Strategy and Architecture closely align with the customer needs. • Create the Service roadmap, reference architecture, book of work, blueprints & technology standards. • Actively lead book of work delivery to ensure successful execution. • Influence and ensure alignment with overall GCTO strategy and reference architecture. • Influence and ensure alignment with the strategies and architectures of related Service and Capabilities. • Handle Service TCO and Pricing.  Credit Suisse maintains a Working Flexibility Policy, subject to the terms as set forth in the Credit Suisse United States Employment Handbook.  Qualifications:  Are you looking forward to work in a fast-paced exciting environment on the leading edge of IT technology and Artificial Intelligence in the financial industry? We can train on the specific technologies, but you should have the following:  Required: • You have at least 7-10 years working in Information Technology, Computer Science, or related fields. • You understand technology and has (or can have a) pulse on industry trends and how they impact Enterprise. • You recognizes the nature and scope of present and future product lines by reviewing product specifications and requirements; appraising new product ideas and/or product or packaging changes. • You have the ability to handle changing priorities, deal with ambiguity and use good judgment in stressful situations. • You experienced handling employees and developing customer relationships across multiple business units. • You have experience in lean, Agile, six-sigma, process improvement. • You have excellent business and financial acumen with past experience in strategic planning, financial planning and analysis, and budgeting. • Excellent analytical skills and effective problem solving.  You should also possess the following industry experience or knowledge: • You have past experience as a Product manager. • You are familiar with JIRA - a plus. • You have AI and Machine Learning experience. • You have track record working within diverse and successful development or design teams.  For more information visit Technology Careers"


# nlp = spacy.load("en_core_web_sm")
# 
# synonyms = []
# doc = nlp(text)
# Nouns = [chunk.text for chunk in doc.noun_chunks]
# print(Nouns)
# first = Nouns[2]
# #print(first)
# synonyms = [] 
# antonyms = [] 
# 
# 
# for syn in wn.synsets(first): 
#     for l in syn.lemmas(): 
#         synonyms.append(l.name()) 
#         if l.antonyms(): 
#             antonyms.append(l.antonyms()[0].name()) 
#   
# #print(synonyms) 
# #print (fuzz.partial_ratio("plant", "Lead automation engineer"))
# 
# ''' 
#     Algorithm for determining job fit:
#     key word in resume in one of the nouns from job description
#     fuzzy compare returns confidence interval higher than 70
#     designation matches job title
#     worked at fortune 500 companies
#     
# '''
     