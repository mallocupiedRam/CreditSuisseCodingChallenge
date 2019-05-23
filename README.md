# CreditSuisseCodingChallenge

Resume Parsing and Matching challenge:

Our solution implements NER, Named Entity Recognition in order to parse the given resume into predetermined subcategories. NER systems 
have been created that use linguistic grammar-based techniques as well as statistical models such as machine learning. Hand-crafted 
grammar-based systems typically obtain better precision, but at the cost of lower recall and months of work by experienced computational 
linguists. Our model uses python's spaCy's statistical NER model with custom labels, and is trained on a dataset containing 20 resumes,
and two subsequent models were generated. For each resume, the two models work in unison to create an aggregate summary of the information
contained in the resumes. Finally, the corresponding labels in the resume are compared to the job description, and a weighted sum is 
computed based on the compatability of the candidate with the position. Finally, the aggregate score is computed and compared for all the
positions available, and the best matched positions are returned to the candidate, urging them to apply. 

Notes: Our data set for the NER model is quite small, a larger data set with more labels should be used, but due to the time taken to 
write the model and create the test data, 20 data points was all we could afford. Furthermore, a better way of computing the weighted
sum that gauges the match between candidate and position should be considered



Design Details:


Data Exploration


Data Preparation

All resumé and job listing data were originally in doc or docx form.  We converted the resumés to strings so that they could then be cleaned up and labelled in a way that is better suited for SpaCy, which is what we used for the NER training.  To do this, we first converted the resumés into txt files, and then from these corresponding text files one by one appended each characer to an empty string.  AFter running these prepared files through our model, we organized the resulting information as a dictionary containing string lists as values for string keys, representing labels and items in the resume corresponding to that label,  Next, these strings were further parsed in train.py and put into appropriate format to run the algorithm.  Similarly, we converted the job descriptions from txt files into strings, but we left the strings as they were and instead used SpaCy to isolate all the nouns in the string.  The final representation of the data before applying our heuristics to determine best candidacy and top picks was the resumés as dictionaries and the job listings as string lists of words in the job description.

Model Training

Our model was trained on 19 different resumés.  This amount was cut short due to time constraints and the need to retrain our data multiple times while working through our code.  For our model, we used SpaCy and an implementation of NER from DataTurks, as seen in train.py.  For the model, we used 10 labels into which to sort and filter resumé items: designation, graduation year, school, skills, name, email address, location, companies worked at, degree, and years of experience.

Acccuracy Statistics

Screenshot of model accuracy

