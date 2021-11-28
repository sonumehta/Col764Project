#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import json
import os
import re
import struct
import numpy as np
import csv
import json
from collections import defaultdict
import math
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import pandas as pd 
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import random
from scipy.sparse import csr_matrix, issparse
import pickle


# In[2]:


os.chdir('C://Users/someh/Desktop/Sem2/IR/Assignment/Assignment2/2020-07-16/')


# In[3]:


ps = PorterStemmer()
stop_words = set(stopwords.words('english')) # | set(string.punctuation)


def def_value():
    return ""

def def_idfvalue():
        return 0


# In[4]:


collection_dir = 'C://Users/someh/Desktop/Sem2/IR/Assignment/Assignment2/2020-07-16/'


# In[5]:


# Generate data for siamesexML training

siameseXMLdata = {}
title_to_cord_id_map = {}
def parsefiles():
    
    
    metadata = pd.read_csv(collection_dir+'/metadata.csv')
    metadata['pmc_json_files'] = metadata['pmc_json_files'].fillna('')
    metadata['title'] = metadata['title'].fillna('')
    metadata['abstract'] = metadata['abstract'].fillna('')
    metadata['pdf_json_files'] = metadata['pdf_json_files'].fillna('')
    metadata = metadata[['cord_uid', 'title', 'abstract', 'pmc_json_files', 'pdf_json_files']].groupby('cord_uid').agg({'title': ''.join,
                                       'abstract': ''.join,
                                         'pmc_json_files':';'.join,
                                         'pdf_json_files': ';'.join}).reset_index()
    N = metadata.shape[0]
   
    for i in range(N):

        row = metadata.iloc[i]
        cord_uid = row['cord_uid']
        title = row['title']
        title_to_cord_id_map[title] = cord_uid
        
        
#         print(cord_uid)
        labels = []
        
        if row['pmc_json_files']:
    
            file_paths = row['pmc_json_files'].replace(";;", " ;").replace(';','').split(' ')
            #print(file_paths)
            if '' in file_paths:
                file_paths.remove('')
            for json_path in file_paths:
                try:
                    
                    with open(json_path) as f_json:
                        full_text_dict = json.load(f_json)
                        #print(full_text_dict['bib_entries'])
                        # grab introduction section from *some* version of the full text
                        #print(full_text_dict['bib_entries'].keys())
                        for key in full_text_dict['bib_entries'].keys():

                            labels.append(full_text_dict['bib_entries'][key]['title'])


                except:
                    pass


        else:
            if row['pdf_json_files']:
                
                file_paths = row['pdf_json_files'].replace(";;", " ;").replace(';','').split(' ')
                if '' in file_paths:
                    file_paths.remove('')
               
                for json_path in file_paths:
                    try:
                        temp_labels = []
                        with open(json_path) as f_json:
                            full_text_dict = json.load(f_json)
                       
                            for key in full_text_dict['bib_entries'].keys():

                                temp_labels.append(full_text_dict['bib_entries'][key]['title'])
                        if(len(temp_labels)> len(labels)):
                            labels = temp_labels

                    except:

                        pass

        labels = [x for x in labels if x !='']
        siameseXMLdata[title]=labels
        
        
         

# Uncomment below to generate data for siamsese XML training
#parsefiles()

# saving data
# with open('C://Users/someh/Desktop/Sem2/IR/ProjectNew/Cord19/title_to_cord_id_map.json','w') as f:
#     json.dump(title_to_cord_id_map, f)
# with open('siameseXMLdata.json','w') as f:
#     json.dump(siameseXMLdata, f)


# with open('C://Users/someh/Desktop/Sem2/IR/ProjectNew/Cord19/title_to_cord_id_map.json','w') as f:
#     json.dump(title_to_cord_id_map, f)


# import json
# with open('siameseXMLdata.json','w') as f:
#     json.dump(siameseXMLdata, f)


# In[6]:


import json
with open('siameseXMLdata.json','r') as f:
    siameseXMLdata = json.load(f)


# In[3]:


with open('C://Users/someh/Desktop/Sem2/IR/ProjectNew/Cord19/title_to_cord_id_map.json','r') as f:
    title_to_cord_id_map=json.load(f)


# In[7]:


labels = set()
for key in siameseXMLdata.keys():
    labels.update(siameseXMLdata[key])
print(len(labels))


ctr = 0
Y_map_ref_to_integer= {}
Y_map_integer_to_ref = {}
for label in labels:
    Y_map_ref_to_integer[label] = ctr
    Y_map_integer_to_ref[ctr] = label
    
    ctr+=1


# In[84]:


# with open('Y_map_ref_to_integer.json','w') as f:
#     json.dump(Y_map_ref_to_integer, f)


# In[18]:


datapoints = [(k, v) for k, v in siameseXMLdata.items()]
random.seed(4)
random.shuffle(datapoints)
#print(relatedfilelines)
trainsize = int(0.8*len(datapoints))

traindata = datapoints[:trainsize]
testdata = datapoints[trainsize:]

print(len(datapoints),len(traindata) ,len(testdata))


# In[19]:


print(len(traindata) ,len(testdata))


# In[20]:


train_title_text = [x[0] for x in traindata]
test_title_text = [x[0] for x in testdata]


# In[14]:


textfile = open("C://Users/someh/Desktop/Sem2/IR/ProjectNew/Cord19_99/train_title_text.txt", "w",encoding="utf-8")

for element in train_title_text:

    textfile.write(str(element) + "\n")

textfile.close()


# In[30]:


train_file_cord_id = open("C://Users/someh/Desktop/Sem2/IR/ProjectNew/Cord19_99/train_cord_id.txt","w")
for i in range(len(train_title_text)):
    train_file_cord_id.write(title_to_cord_id_map[train_title_text[i]]+"\n")
    
train_file_cord_id.close()   


# In[15]:


textfile = open("C://Users/someh/Desktop/Sem2/IR/ProjectNew/Cord19_99/test_title_text.txt", "w",encoding="utf-8")

for element in test_title_text:

    textfile.write(str(element) + "\n")

textfile.close()


# In[31]:


test_file_cord_id = open("C://Users/someh/Desktop/Sem2/IR/ProjectNew/Cord19_99/test_cord_id.txt","w")
for i in range(len(test_title_text)):
    test_file_cord_id.write(title_to_cord_id_map[test_title_text[i]]+"\n")
    
test_file_cord_id.close()   


# In[39]:

# Writing the label files for both train and test

def write_to_file(filename, relateddict):
    #file = open("trn_X_Y.txt", "w")
    file = open(filename, "w")
    file.write(str(len(relateddict)) + " " + str(len(Y_map_ref_to_integer.keys()))+ "\n")
    for tuplet in relateddict:
        labels = tuplet[1]
        #print(tuplet[0])
        string_to_write = ""
        #print(labels)
        
        labels_to_be_written=set()
        for label in labels:
            labels_to_be_written.add(Y_map_ref_to_integer[label])
            
        labels_to_be_written=np.sort(list(labels_to_be_written))
        #print(labels_to_be_written)
        for label in labels_to_be_written:
            
            string_to_write+= str(label)+":1 "
            
        #print(string_to_write)
        file.write(string_to_write+"\n")   
    file.close()
    
write_to_file('C://Users/someh/Desktop/Sem2/IR/ProjectNew/Cord19_99/trn_X_Y.txt', traindata)
    
write_to_file("C://Users/someh/Desktop/Sem2/IR/ProjectNew/Cord19_99/tst_X_Y.txt", testdata)


# In[34]:

# tokenizing the data and generating features

stop = list(stopwords.words('english'))
def tokenize(raw):
    return [w.lower() for w in word_tokenize(raw) if w.isalpha()]

class StemmedTfIdfVectorizer(TfidfVectorizer):
    ps = PorterStemmer()
    def build_analyzer(self):
        
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc:(ps.stem(w) for w in analyzer(doc))



# In[41]:


features_data = set()
for key in siameseXMLdata.keys():
    features_data.add(key)
    features_data.update(siameseXMLdata[key])
features_data=list(features_data)
print(len(features_data))


# In[42]:


def get_vectorizer(features_data):
    vectorizer = StemmedTfIdfVectorizer(
    tokenizer=tokenize, 
    analyzer="word", 
    stop_words='english', 
    ngram_range=(1,1), 
    #min_df=3    # limit of minimum number of counts: 3
)
    
    
    #vectorizer = StemmedTfIdfVectorizer()
    vectorizer.fit(features_data)
    featurenames = vectorizer.get_feature_names()
    #print((featurenames))
    #print(vectorizer.transform(['yield yellow war wuhan','covid efficacy europe infection']))
    return vectorizer
    
def gen_features(vectorizer, data):
    result = vectorizer.transform(data)
    return result

vectorizer = get_vectorizer(features_data)


# In[43]:


with open('C://Users/someh/Desktop/Sem2/IR/ProjectNew/Cord19_99/vectorizer.pk', 'wb') as fin:
    pickle.dump(vectorizer, fin)


# In[35]:


vectorizer = pickle.load(open('C://Users/someh/Desktop/Sem2/IR/ProjectNew/Cord19_99/vectorizer.pk','rb'))


# Writing the features files for both train and test

train_sparse_matrix = gen_features(vectorizer,train_title_text)
print('train done')
test_sparse_matrix = gen_features(vectorizer,test_title_text)
print('test done')
label_sparse_matrix = gen_features(vectorizer, labels)
print('label done')


# In[39]:


def write_sparse_file(X, filename, header=True):
    """Write sparse label matrix to text file (comma separated)
    Header: (#users, #labels)
    
    Arguments
    ----
    X: sparse matrix
        data to be written
    filename: str
        output file
    header: bool, default=True
        write header or not
    """
    if not isinstance(X, csr_matrix):
        X = X.tocsr()
    X.sort_indices()
    with open(filename, 'w') as f:
        if header:
            print("%d %d" % (X.shape[0], X.shape[1]), file=f)
        for y in X:
            idx = y.__dict__['indices']
            val = y.__dict__['data']
            sentence = ' '.join(['{}:{}'.format(x, v)
                                 for x, v in zip(idx, val)])
            print(sentence, file=f)





write_sparse_file(train_sparse_matrix, "C://Users/someh/Desktop/Sem2/IR/ProjectNew/Cord19_99/trn_X_Xf.txt")
write_sparse_file(test_sparse_matrix, "C://Users/someh/Desktop/Sem2/IR/ProjectNew/Cord19_99/tst_X_Xf.txt")
write_sparse_file(label_sparse_matrix, "C://Users/someh/Desktop/Sem2/IR/ProjectNew/Cord19_99/lbl_X_Xf.txt")


