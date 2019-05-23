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
