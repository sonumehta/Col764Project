import torch
import pickle
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import json
import numpy as np
from bs4 import BeautifulSoup
import pytrec_eval

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


from scipy.sparse import csr_matrix, issparse
from collections import defaultdict

import numpy as np
import os
import sys
import torch
from xclib.data import data_utils
import torch.nn.functional as F
from xclib.utils.sparse import normalize

def compute_representation(x, embedding):
    """Compute the dense representation for a given item
        - tf-idf weighted token embeddings for now
    """
    x = normalize(x) @ embedding  # each instance is normalized to unit norm
    x = F.gelu(torch.from_numpy(x)).numpy()
    return normalize(x)

def tokenize(raw):
    return [w.lower() for w in word_tokenize(raw) if w.isalpha()]

class StemmedTfIdfVectorizer(TfidfVectorizer):
    ps = PorterStemmer()
    def build_analyzer(self):
        
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc:(ps.stem(w) for w in analyzer(doc))

def get_cord_id_from_index(data, index):    
    return title_to_cord_id_map[data[index][:-1]]
    
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
    
    
model_dir = '/data/someh/models/SiameseXML/Astec/Cord19_99/v_0_21/'
cord19_dir = '/data/someh/data/Cord19_99/'
cord19_evaluator = '/data/someh/data/Cord19_Evaluator/'
results_file = 'SiameseXML_results.txt'

Zmatrix = torch.load(os.path.join(model_dir, 'Z.pkl'))
token_embeddings = Zmatrix['net']['transform.embeddings.weight'].cpu().numpy()[1:]
stop = list(stopwords.words('english'))
ps = PorterStemmer()
vectorizer = pickle.load(open(os.path.join(cord19_dir, 'vectorizer.pk'),'rb'))


train_file = open(os.path.join(cord19_dir, 'train_title_text.txt'),'r', encoding = 'utf-8')
train_title_text = train_file.readlines()
train_file.close()

test_file = open(os.path.join(cord19_dir, 'test_title_text.txt'),'r', encoding = 'utf-8')
test_title_text = test_file.readlines()
test_file.close()

trn_fts = data_utils.read_sparse_file(os.path.join(cord19_dir, 'trn_X_Xf.txt'))
tst_fts = data_utils.read_sparse_file(os.path.join(cord19_dir, 'tst_X_Xf.txt'))
trn_doc_embeddings = compute_representation(trn_fts, token_embeddings)
tst_doc_embeddings = compute_representation(tst_fts, token_embeddings)
with open('trn_doc_embeddings.npy', 'wb') as f:

    np.save(f, trn_doc_embeddings)
    
with open('tst_doc_embeddings.npy', 'wb') as f:

    np.save(f, tst_doc_embeddings)
with open(os.path.join(cord19_evaluator, 'title_to_cord_id_map.json')) as f_json:
    title_to_cord_id_map = json.load(f_json)

query_dict = get_queries_for_eval(os.path.join(cord19_evaluator, 'topics-rnd5.xml'))


output_file_to_be_written = open(os.path.join(cord19_dir, results_file),"w")
for query_number in query_dict.keys():

    query = query_dict[query_number]
    print(query_number, query)
    query_vector = vectorizer.transform([query])
    query_embedding = compute_representation(query_vector, token_embeddings)
    similarities_train = cosine_similarity(query_embedding, trn_doc_embeddings)
    similarities_test = cosine_similarity(query_embedding, tst_doc_embeddings)
    doc_scores = {}
    for i in range(len(similarities_train[0])):
        
        doc_scores[ ] = similarities_train[0][i]
        
    for i in range(len(similarities_test[0])):
        
        doc_scores[get_cord_id_from_index(test_title_text, i)] = similarities_test[0][i]
    
    doc_scores = {k: v for k, v in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)}
   
    ctr_file = 0
    score_dict = {}

    rank = 1
    index = 0
    for doc_id in doc_scores.keys():
        row = [query_number, 'Q0', doc_id, rank, doc_scores[doc_id], "runid1" ]
        rank+=1
        index+=1
        output_file_to_be_written.write(" ".join([str(x) for x in row]))
        output_file_to_be_written.write("\n")
    
output_file_to_be_written.close()

qrel_file = os.path.join(cord19_evaluator, 'qrels-covid_d5_new.txt')

get_scores(qrel_file, os.path.join(cord19_dir, results_file))

