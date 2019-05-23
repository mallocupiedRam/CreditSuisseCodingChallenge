import sys
from parseResume import *
from parseDescription import *
import time
from word2vecStuffs import *

def generateTopJobs(filename):
    parsed = createParsed(filename)
    return getTopJobs(parsed)

#converts doc to strings
def genJobString(filename):    
    parsed = filename.split(".")
    extension = parsed[len(parsed) -1]
    parsed = parsed[:-1]
    s = "."
    file = s.join(parsed)
    global_file = file
    
    #checks for right extension
    if (extension == "docx" or extension == "doc"):
        text = docx2txt.process(filename)
        final = ""
        #omits non-ascii characters
        for char in text:
            if (32 <= ord(char) <= 126 or ord(char) == 10):
                final += char
    return final

#runs the commandline prompt
def main():
    print('''If you are a applicant, provide a path to your resume to see the top
          jobs you are the best fit for. If you are a client, give a path to the 
          position you want the best candidates for with a -c option''')
    args = sys.argv[1:]
    client = False
    if (len(args) > 2): 
        print("Invalid Argument")
    if (len(args) == 2 and args[1] != '-c'):
        print("Invalid option, please provide -c if you are a client")
    
    if (len(args) == 2):
        client = True
    
    if (not client):
        #generates top 3 jobs if path provided is to a resume 
        #(candidate looking for best suited jobs)
        print(generateTopJobs(args[0]))
    else:
        #provides best candidates suited to a job
        job_desc = genJobString(args[0])
        output_best("Resumes/text/", job_desc)
        
    return 0
main()
    
    
    

