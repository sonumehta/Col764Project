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
from rank_bm25 import BM25Okapi
import pytrec_eval


# In[2]:


ps = PorterStemmer()
stop_words = set(stopwords.words('english')) # | set(string.punctuation)


def def_value():
    return ""

def def_idfvalue():
        return 0


# In[3]:


def get_file_content(collection_dir):
    ps = PorterStemmer()
    file_content = {}
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
    #N = 10
    for i in range(N):
        if(i%10000 ==0 ):
            print(i)
        row = metadata.iloc[i]
        cord_uid = row['cord_uid']
        title = row['title']
        abstract = row['abstract']
        #print(cord_uid)
        body_text = []

        
        if row['pmc_json_files']:

            file_paths = row['pmc_json_files'].replace(";;", " ;").replace(';','').split(' ')
            #print(file_paths)
            if '' in file_paths:
                file_paths.remove('')
            for json_path in file_paths:
                try:

                    with open(json_path) as f_json:
                        full_text_dict = json.load(f_json)

                        # grab introduction section from *some* version of the full text
                        for paragraph_dict in full_text_dict['body_text']:
                            paragraph_text = paragraph_dict['text']
                            body_text.append(paragraph_text)
                except:
                    pass

        else:
            if row['pdf_json_files']:

                file_paths = row['pdf_json_files'].replace(";;", " ;").replace(';','').split(' ')
                if '' in file_paths:
                    file_paths.remove('')

                for json_path in file_paths:
                    try:

                        with open(json_path) as f_json:
                            full_text_dict = json.load(f_json)

                            # grab introduction section from *some* version of the full text
                            for paragraph_dict in full_text_dict['body_text']:
                                paragraph_text = paragraph_dict['text']

                                body_text.append(paragraph_text)
                    except:

                        pass

        combined_string = title + " " + abstract + " " +" ".join(body_text)
        word_tokens = word_tokenize(re.sub(r'[^a-zA-Z0-9]',' ',combined_string.lower()))
        #tokens_without_sw = [word for word in word_tokens if not word in stopwords]
        
        processed = [ps.stem(w) for w in word_tokens if w not in stop_words]

            
        #print(combined_string)
        file_content[cord_uid] = processed
    return file_content


# In[4]:


def get_processed_text(text):
    word_tokens= word_tokenize(re.sub(r'[^a-zA-Z0-9]',' ',text.lower()))
    processed = [ps.stem(w) for w in word_tokens if w not in stop_words]
    return processed

def get_queries_for_eval(query_file):
    query_dict = {}
    covid_topics_file = open(query_file, "r")
    contents = covid_topics_file.read()
    soup = BeautifulSoup(contents, 'html.parser')
    topics_list = soup.find_all('topic')
    for topic in topics_list:
        query_dict[topic['number']] = topic.query.text

    return query_dict

def get_scores(qrel_file, outputfile):

    def def_idfvalue():
            return 0
    topic_count = 50

    def get_true_relevant_docs():
        with open(qrel_file) as file:
            lines = file.readlines()

        true_relevant_docs = defaultdict(list)
        true_relevant_scores = defaultdict()

        for line in lines:
            tokens = re.split(' |\n', line)

            true_relevant_docs[tokens[0]].append(tokens[2])
            true_relevant_scores[(tokens[0],tokens[2])]=int(tokens[3])

        result = {}
        for i in range(topic_count):
            temp = {}
            for doc in true_relevant_docs[str(i+1)]:
                temp[doc] = true_relevant_scores[(str(i+1), doc)]
            result[str(i+1)] = temp
       
        return result

    def get_updated_scores():
       
       
        with open(outputfile) as file:
            lines = file.readlines()
        true_relevant_docs = defaultdict(list)
        true_relevant_scores = defaultdict()

        for line in lines:
            try:
                tokens = re.split(' |\n', line)
                true_relevant_docs[tokens[0]].append(tokens[2])
                true_relevant_scores[(tokens[0],tokens[2])]=float(tokens[4])
            except:
                print(line)

        result = {}
        for i in range(topic_count):
            temp = {}
            for doc in true_relevant_docs[str(i+1)]:
                temp[doc] = true_relevant_scores[(str(i+1), doc)]
            result[str(i+1)] = temp
        return result


    qrel = get_true_relevant_docs()
    updated = get_updated_scores()
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {'map', 'ndcg_cut_5', 'ndcg_cut_10', 'ndcg_cut_20'})

    metrics_dict = evaluator.evaluate(updated)
    avg_ndcg_5=0
    avg_ndcg_10=0
    avg_ndcg_20=0
    avg_map=0
    #print(metrics_dict)
    for query in metrics_dict:
        avg_map+=metrics_dict[query]['map']
        avg_ndcg_5+=metrics_dict[query]['ndcg_cut_5']
        avg_ndcg_10+=metrics_dict[query]['ndcg_cut_10']
        avg_ndcg_20+=metrics_dict[query]['ndcg_cut_20']


    avg_map_q = avg_map/topic_count
    avg_ndcg_5_q = avg_ndcg_5/topic_count
    avg_ndcg_10_q = avg_ndcg_10/topic_count
    avg_ndcg_20_q = avg_ndcg_20/topic_count

    print("AVerage MAP",avg_map/topic_count)
    print("Average NDCG@5", avg_ndcg_5/topic_count)
    print("Average NDCG@10", avg_ndcg_10/topic_count)
    print("Average NDCG@20", avg_ndcg_20/topic_count)

    print(avg_map_q,avg_ndcg_5_q, avg_ndcg_10_q, avg_ndcg_20_q)

    return [avg_map_q,avg_ndcg_5_q, avg_ndcg_10_q, avg_ndcg_20_q]


# In[5]:


data_dir = 'C://Users/someh/Desktop/Sem2/IR/ProjectNew/Bm25data/'
qrel_file = os.path.join(data_dir, 'qrels-covid_d5_new.txt')
collection_dir = 'C://Users/someh/Desktop/Sem2/IR/Assignment/Assignment2/2020-07-16/'


# In[ ]:


# Uncomment this to generate the required data
# collection_dir = 'C://Users/someh/Desktop/Sem2/IR/Assignment/Assignment2/2020-07-16/'
# file_content= get_file_content(collection_dir)
# with open(os.path.join(data_dir, 'filecontent.json'),'w') as f:
#     json.dump(file_content)


# In[6]:


with open(os.path.join(data_dir, 'filecontent.json'),'r') as f:
    filecontent = json.load(f)


# In[8]:


file_content_list = [x for x in filecontent.values()]


# In[9]:


query_dict = get_queries_for_eval(os.path.join(data_dir, 'topics-rnd5.xml'))


# In[43]:


# Tuning Hyperparameter for BM25
k1_list = np.linspace(0.5, 2, 10)
b_list = np.linspace(0, 1, 10)
list_of_cord_ids = list(filecontent.keys())
x, y = np.meshgrid(k1_list, b_list)
# print(x,y)
best_score = {}
best_param = {}
metrics = {}

counter = 0
for k1, b in zip(x.reshape(-1), y.reshape(-1)):
    counter+=1
#     k1 = float(k1)
#     b = float(b)
    print(k1,b)
    bm25 = BM25Okapi(file_content_list, k1= k1, b=b)
    
    output_file = os.path.join(data_dir, 'BM25_results'+str(counter))
    output_file_to_be_written = open(output_file,"w")

    for query_number in query_dict.keys():

        query = query_dict[query_number]
        #print(query_number, query)
        tokenized_query = get_processed_text(query)
        doc_scores = bm25.get_scores(tokenized_query)

        rank = 1
        index = 0
        for score in doc_scores:
            row = [query_number, 'Q0', list_of_cord_ids[index], rank, score, "runid1" ]
            rank+=1
            index+=1
            output_file_to_be_written.write(" ".join([str(x) for x in row]))
            output_file_to_be_written.write("\n")
    output_file_to_be_written.close()
    
    
#     revals = get_scores(qrel_file,output_file)
#     df_reval = pd.DataFrame(revals).T
#     metric = dict(df_reval.mean(axis=0))

#     for m in metric:
#         best = best_score.get(m, 0)
#         if best < metric[m]:
#             best_score[m] = metric[m]
#             best_param[m] = (a, b)

#         l = metrics.setdefault(m, [])
#         l.append(metric[m])

    


# In[ ]:


# Code to evalaute the performance of BM25

topic_count = 50
def def_idfvalue():
            return 0
    

def get_true_relevant_docs():
    with open(qrel_file) as file:
        lines = file.readlines()

    true_relevant_docs = defaultdict(list)
    true_relevant_scores = defaultdict()

    for line in lines:
        tokens = re.split(' |\n', line)

        true_relevant_docs[tokens[0]].append(tokens[2])
        true_relevant_scores[(tokens[0],tokens[2])]=int(tokens[3])

    result = {}
    for i in range(topic_count):
        temp = {}
        for doc in true_relevant_docs[str(i+1)]:
            temp[doc] = true_relevant_scores[(str(i+1), doc)]
        result[str(i+1)] = temp

    return result



qrel = get_true_relevant_docs()    
    
def get_scores(outputfile):

    def get_updated_scores():
        with open(outputfile) as file:
            lines = file.readlines()
        true_relevant_docs = defaultdict(list)
        true_relevant_scores = defaultdict()

        for line in lines:
            try:
                tokens = re.split(' |\n', line)
                true_relevant_docs[tokens[0]].append(tokens[2])
                true_relevant_scores[(tokens[0],tokens[2])]=float(tokens[4])
            except:
                print(line)

        result = {}
        for i in range(topic_count):
            temp = {}
            for doc in true_relevant_docs[str(i+1)]:
                temp[doc] = true_relevant_scores[(str(i+1), doc)]
            result[str(i+1)] = temp
        return result

    updated = get_updated_scores()
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, {'map', 'ndcg_cut_5', 'ndcg_cut_10', 'ndcg_cut_20'})

    metrics_dict = evaluator.evaluate(updated)
    avg_ndcg_5=0
    avg_ndcg_10=0
    avg_ndcg_20=0
    avg_map=0
    #print(metrics_dict)
    for query in metrics_dict:
        avg_map+=metrics_dict[query]['map']
        avg_ndcg_5+=metrics_dict[query]['ndcg_cut_5']
        avg_ndcg_10+=metrics_dict[query]['ndcg_cut_10']
        avg_ndcg_20+=metrics_dict[query]['ndcg_cut_20']


    avg_map_q = avg_map/topic_count
    avg_ndcg_5_q = avg_ndcg_5/topic_count
    avg_ndcg_10_q = avg_ndcg_10/topic_count
    avg_ndcg_20_q = avg_ndcg_20/topic_count

    print("AVerage MAP",avg_map/topic_count)
    print("Average NDCG@5", avg_ndcg_5/topic_count)
    print("Average NDCG@10", avg_ndcg_10/topic_count)
    print("Average NDCG@20", avg_ndcg_20/topic_count)

    print(avg_map_q,avg_ndcg_5_q, avg_ndcg_10_q, avg_ndcg_20_q)

    return [avg_map_q,avg_ndcg_5_q, avg_ndcg_10_q, avg_ndcg_20_q]
    
k1_list = np.linspace(0.5, 2, 10)
b_list = np.linspace(0, 1, 10)

x, y = np.meshgrid(k1_list, b_list)
# print(x,y)
best_score = {}
best_param = {}
metrics = {}

counter = 1
output_file = os.path.join(data_dir, 'BM25_results'+str(counter))
print(output_file)
revals = get_scores(output_file)
df_reval = pd.DataFrame(revals).T
metric = dict(df_reval.mean(axis=0))
print(metric)

counter = 0
for k1, b in zip(x.reshape(-1), y.reshape(-1)):
    counter+=1
    print(counter)
    
    output_file = os.path.join(data_dir, 'BM25_results'+str(counter))
        
    revals = get_scores(output_file)
    df_reval = pd.DataFrame(revals).T
    metric = dict(df_reval.mean(axis=0))

    for m in metric:
        best = best_score.get(m, 0)
        if best < metric[m]:
            best_score[m] = metric[m]
            best_param[m] = (k1, b)

        l = metrics.setdefault(m, [])
        l.append(metric[m])
        

print(metrics)
print(best_score)
print(best_param)


# In[13]:


#Saving the model with the best score
k1 = 2.0 
b= 0.2222222222222222
print(k1,b)
bm25 = BM25Okapi(file_content_list, k1= k1, b=b)
with open('C://Users/someh/Desktop/Sem2/IR/ProjectNew/BM25data/bm25.pk', 'wb') as fin:
    pickle.dump(bm25, fin)


# In[14]:





# In[10]:


# print the metric values for different set of hyperparameters
metrics = {0: [0.23521757887501696, 0.23718537632676612, 0.23826746462536996, 0.23865523800213, 0.23856557845423537, 0.238101356850505, 0.2372796465783967, 0.2361470586696906, 0.23514757513068546, 0.23403547344788225, 0.24348034921166467, 0.2456003620847472, 0.2468337235444508, 0.24711131981878748, 0.2469230292173913, 0.24626798230243885, 0.2455186059768033, 0.2444737363189768, 0.24338989854981805, 0.24207812022248465, 0.24673522412648488, 0.24923888170929395, 0.2505144526166324, 0.2508375539639804, 0.2507646374519384, 0.2501775669367911, 0.24933731529076678, 0.24839281234248123, 0.24712340135739147, 0.24596886551744707, 0.2485132489620442, 0.2511665391437044, 0.25243547665786215, 0.2529905082394093, 0.25287119012598047, 0.252499560147894, 0.2517636711408629, 0.25081067047900474, 0.24969306576010733, 0.2486359560986314, 0.24949883617524982, 0.25245776871516834, 0.25366744949902603, 0.25422806511898816, 0.2542958012814092, 0.25385113340120713, 0.2531589861468618, 0.25235657191314903, 0.25139008810388874, 0.25033803026794277, 0.2493277024980401, 0.2520380848209449, 0.2531859067205525, 0.2538402814713053, 0.2540579302699637, 0.2537088423397364, 0.25297219984760416, 0.2521415766144272, 0.25127146516989207, 0.25030269524318777, 0.24744151928026611, 0.2500219328433582, 0.25118134485654353, 0.25150502574396894, 0.2514019343711612, 0.2510278063810903, 0.2502609147197278, 0.24948308716566825, 0.24835103232639585, 0.24735145631549227, 0.2425825244584142, 0.24486271740840312, 0.24577519631170475, 0.24582629325080316, 0.2452821128993609, 0.24427716610951472, 0.24337865608426995, 0.2423858932304351, 0.24121203920108752, 0.23995277393673267, 0.2356935512991156, 0.2365676689231075, 0.23620354039365052, 0.23494956784872334, 0.23333955866332848, 0.23140834513678424, 0.22927001697351707, 0.2271824948367892, 0.22519559581123066, 0.22323798330462732, 0.22727258870908704, 0.20998400370763992, 0.2202892786815018, 0.21151650945036313, 0.2060137045163534, 0.1987888332068745, 0.19898059734849, 0.194654586794525, 0.19024994652333724, 0.18570788963378349], 1: [0.6381032216703182, 0.6429555072214813, 0.6394292762924612, 0.6354669505321862, 0.6248139448181728, 0.6267588672195012, 0.619975663114029, 0.61490042962281, 0.6126259380991805, 0.6122701978317745, 0.6572717367002288, 0.6538216598266938, 0.6559956604088667, 0.6541207595324353, 0.6464975762566553, 0.6407754650024121, 0.6468173420403, 0.6458488642344968, 0.6487080833126848, 0.641293844129941, 0.6537272991260259, 0.6604903691436901, 0.6564214244909695, 0.6567710345175868, 0.6514625200473607, 0.6464475979258744, 0.6399256847832479, 0.6416817971793485, 0.624757927140295, 0.6291198435477681, 0.664089078740529, 0.6598391999786104, 0.6552561698902162, 0.6519467589502623, 0.6449887509763811, 0.6442876011907209, 0.6401266643763612, 0.6340226394198072, 0.6258810770164984, 0.6230200213893768, 0.6629317909294978, 0.6601066188924456, 0.6549990817255332, 0.6562989652025708, 0.6543698831971955, 0.6430943366061289, 0.6316556237444437, 0.6243540428012423, 0.6252622092373128, 0.622214180153897, 0.6593732586145835, 0.6529013238027379, 0.6579905634307761, 0.657351561563782, 0.6523348028933619, 0.6389815415736164, 0.6239221484783508, 0.6180812510335763, 0.6209142966768593, 0.6133252513359071, 0.6492288286894428, 0.6450752357810184, 0.6460998221583912, 0.6541731064022451, 0.6478863107875596, 0.6295419345320034, 0.6220637108816596, 0.6232027920475083, 0.622057580640871, 0.6185393165015731, 0.6000524400533013, 0.6064719371016452, 0.6170121936824243, 0.6096966088922033, 0.6084388301397157, 0.6041091242244447, 0.5968015039341465, 0.5996004116426018, 0.6097210343820667, 0.6076012149288378, 0.5106109032792628, 0.5163774855072853, 0.510532294370656, 0.5055332126451229, 0.5112133100685079, 0.5094310242375025, 0.5098952199462878, 0.5099195568322589, 0.5145610617414522, 0.5193390297860637, 0.49275260018574224, 0.47772734502530434, 0.47283058850036186, 0.4625193192067351, 0.45318449450863374, 0.43852898575242094, 0.43715662360756496, 0.4278016648216554, 0.4200700261044962, 0.4183742250781281], 2: [0.6096791620279793, 0.6167015473652656, 0.6129646433314618, 0.603440975179929, 0.5984028395283106, 0.595843524795811, 0.5892241775423096, 0.5897950920889787, 0.5865101201934513, 0.5862472366155526, 0.6347651212687117, 0.6329315359805059, 0.6286582957173805, 0.6263573162992307, 0.6202591262530428, 0.6159701479234618, 0.6144800921940382, 0.6105951755137212, 0.6111972619938724, 0.5969770360663095, 0.6329324176927136, 0.633233971097907, 0.6322361684159895, 0.6259783280664042, 0.6198863198155314, 0.6187366825570626, 0.6136656374453188, 0.6118562014274732, 0.5981174762742871, 0.593480258247346, 0.6415445462021226, 0.6393667798853698, 0.6348844174239637, 0.6273157529090019, 0.6243556584156247, 0.6219772654804993, 0.6138282909083872, 0.606473505965179, 0.6001463804359852, 0.594578070434983, 0.6436107893402233, 0.6423355859176441, 0.6327072253555581, 0.6295144378500929, 0.6243951470297922, 0.6167455960859739, 0.6100534212823676, 0.6028638707073812, 0.5998584347012011, 0.5985229561746468, 0.6412184773708711, 0.6417741207594032, 0.6322932299740205, 0.6266366183521158, 0.6269619361159251, 0.6206737332606886, 0.616457839752817, 0.611503121311254, 0.6025245804914985, 0.6004257732183041, 0.6312339821603282, 0.6282574872026704, 0.6229482145756022, 0.6222978492659054, 0.6182429149580948, 0.616604413997272, 0.6148766482852727, 0.6133901504834602, 0.6059242751219422, 0.6022628576947998, 0.5905495943842836, 0.5924363299639139, 0.5972135005682364, 0.598535105770628, 0.598834116638204, 0.5941391194254215, 0.5953682578525077, 0.5974088062183257, 0.5903574167422484, 0.5861991731044596, 0.5123949323955581, 0.5166309214290373, 0.5184465272750173, 0.5178026398194484, 0.5169143756787783, 0.5126396486544501, 0.5089905073291016, 0.5064060387282442, 0.5066801945941871, 0.5033451196414671, 0.4874963258367833, 0.4780202065275006, 0.471174031689868, 0.45504720499694573, 0.4382930590010983, 0.4286831674090702, 0.42728742270159725, 0.42654104155208244, 0.41764699055609533, 0.41374640794494466], 3: [0.5778159490922428, 0.571759411339769, 0.5684677472881284, 0.5663958254751076, 0.5648491884732078, 0.557312371783537, 0.5508383157944873, 0.5445970411712768, 0.5443051419359785, 0.5425397632279747, 0.5950000186194713, 0.5900616859214146, 0.5881345770742691, 0.5830956805443372, 0.5794500974655602, 0.5761752033040647, 0.572681552343792, 0.5647801840235287, 0.5592741168835318, 0.5540500901234567, 0.6057517721779073, 0.6067469023548333, 0.6014867797733595, 0.5962807623345008, 0.5909084592446918, 0.5816840069998238, 0.5755769086179623, 0.5706530034427854, 0.5615475669544646, 0.5566219732588801, 0.6106127451491314, 0.6082908885296631, 0.6051597328484111, 0.5965048599446754, 0.5963875909499274, 0.5912769761453183, 0.583923150272153, 0.5784498410430033, 0.5704921990459265, 0.5662694450201468, 0.6058572874629381, 0.6023388921815102, 0.5949137933246318, 0.5929343407783034, 0.5908477988476117, 0.5842463150323713, 0.5829397078267529, 0.5790337147797616, 0.5738281042406047, 0.568180922719995, 0.6024208401083075, 0.5966759745043695, 0.5948023090529163, 0.5893458266146674, 0.5900440454491505, 0.5868338991923623, 0.5802707330159753, 0.5758303166083403, 0.5729265419999574, 0.5703473215040558, 0.5871920043583302, 0.587614750739986, 0.5862015301348883, 0.585734000077553, 0.5838329409165989, 0.580397621588221, 0.579327089986346, 0.57197937059041, 0.5665751290117064, 0.5606858242107645, 0.5582333379007448, 0.5596955528802866, 0.5577369529627457, 0.5567294173232344, 0.5563423812975128, 0.5539014143994906, 0.5540314018793238, 0.5522501014108901, 0.5517408060484552, 0.5505320332007431, 0.5092188859857084, 0.5079711781706634, 0.504060245164239, 0.4968827543283812, 0.4875138800537788, 0.4796878887180423, 0.4807879129599663, 0.4779144553704613, 0.47507520154324356, 0.47230613265754956, 0.47340402904512197, 0.4495379970244317, 0.44220780940959176, 0.42641260913422385, 0.41922934609859575, 0.4032856963394126, 0.4026156364128507, 0.4030464491451382, 0.3964636566489149, 0.38975430119155136]}
import numpy as np
import matplotlib.pyplot as plt
k1_list = np.linspace(0.5, 2, 10)
b_list = np.linspace(0, 1, 10)

map_of_metrics = {0:'map', 1:'ndcg@5', 2:'ndcg@10',3:'ndcg@20'}
def scatter_plot(x, y, z, rows=2, figsize=(20, 10)):
    
    num = len(metrics)
    cols = int(np.ceil(num/rows))
    keys = list(z.keys())
    
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    for r in range(rows):
        for c in range(cols):
            
            n = r*rows + c
            if n < num:
                
                im = ax[r, c].scatter(x, y, c=z[keys[n]], cmap='viridis')
                
                ax[r, c].set_title(map_of_metrics[keys[n]])
                ax[r, c].set_xlabel('k1')
                ax[r, c].set_ylabel('b')
                fig.colorbar(im, ax=ax[r, c])
            else:
                ax[r, c].axis('off')


# for m in best_score:
#     print(f"Best {m}  : {best_score[m]:.2f}, alpha : {best_param[m][0]:.2f}, beta : {best_param[m][1]:.2f}")
x, y = np.meshgrid(k1_list, b_list)
scatter_plot(x.reshape(-1), y.reshape(-1), metrics)
