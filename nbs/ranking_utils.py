import os
import re
import numpy as np
import pandas as pd
import csv
import json
from collections import defaultdict
from bs4 import BeautifulSoup as bs
#import pytrec_eval
import time

#import krovetz
import pickle
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

from scipy.sparse import csr_matrix
from scipy import sparse

plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#Krovetz = krovetz.PyKrovetzStemmer()

"""
read qrel file
"""
def read_qrels(qrel_file):
    qrels = {}
    with open(qrel_file) as file:
        for line in file:
            line = line.strip('\n').strip(' ')
            if not line:
                break
            topic, _, corduid, rel = line.split()
            if topic not in qrels:
                qrels[topic] = {}
            qrels[topic][corduid.strip(' ')] = int(rel)
    return qrels

def doc_gt_score(qrel, corduids):
    gt_score = {}
    for corduid in corduids:
        #gt_score[corduid] = qrel.get(corduid, 0)
        #debug
        if corduid in qrel:
            gt_score[corduid] = qrel[corduid]
        #debug
    return gt_score


"""
parses t40 top100 file
"""
def parser_top_file(top_file):
    top_scores = {}
    with open(top_file) as file:
        for line in file:
            line = line.strip('\n').strip(' ')
            if not line:
                break

            topic, _, corduid, _, score, _ = line.split(' ')
            if topic not in top_scores:
                top_scores[topic] = {}

            top_scores[topic][corduid.strip(' ')] = float(score)

    return top_scores


"""
get meta information
"""

def get_meta_information(query_file, top_file, metafile):
    """
    read queries
    """
    queries_text = get_queries(query_file, 'query')

    """
    reading t40 top100 files
    """
    top_scores = parser_top_file(top_file)

    """
    reading all the meta data
    """
    all_top_corduid = set()
    for top in top_scores.values():
        all_top_corduid.update(list(top.keys()))
    all_top_corduid = list(all_top_corduid)

    top_metadata = exact_documents_metadata(metafile,
                                               all_top_corduid)

    return queries_text, top_scores, top_metadata




"""
get document metadata
"""
def all_documents_metadata(metafile):
    docs_meta = defaultdict(list)

    with open(metafile) as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            docs_meta[row['cord_uid']].append(row)
    return docs_meta


def exact_documents_metadata(metafile, docs_id):
    docs_meta = defaultdict(list)

    with open(metafile) as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            if row['cord_uid'] in docs_id:
                docs_meta[row['cord_uid']].append(row)
    return docs_meta

"""
read queries
"""
def get_queries(query_file, query_type='query'):
    with open(query_file, "r") as file:
        content = file.read()

    bs_content = bs(content, "html.parser")
    topics = bs_content.find_all('topic')

    topic_text = {}
    for topic in topics:
        number = topic['number']
        text = topic.find(query_type).get_text().strip()
        topic_text[number] = text

    return topic_text


"""
document text extraction
"""
def read_doc_body(filename):
    doc_text = []
    doc_section = []
    with open(filename) as f_json:
        full_text_dict = json.load(f_json)

        for paragraph_dict in full_text_dict['body_text']:
            paragraph_text = paragraph_dict['text']
            section_name = paragraph_dict['section']
            if len(paragraph_text) > 0:
                doc_text.append(paragraph_text)
                doc_section.append(section_name)
    return doc_text, doc_section


def read_pmc_text(doc_metadatas, doc_dir):
    pmc_text = ''
    for meta in doc_metadatas:
        pmc_file = meta['pmc_json_files']
        if len(pmc_file) > 0:
            title = meta['title']
            abstract = meta['abstract']
            pmc_file = f"{doc_dir}/{pmc_file}"
            body_text, body_section = read_doc_body(pmc_file)
            body = " ".join(body_text)
            pmc_text = " ".join([title, abstract, body])
    return pmc_text

def read_pdf_text(doc_metadatas, doc_dir):
    best_pdf_text = ''
    best_pdf_len = 0
    for meta in doc_metadatas:
        pdf_files = meta['pdf_json_files'].split('; ')
        for pdf_file in pdf_files:
            body = ''
            if len(pdf_file) > 0:
                pdf_file = f"{doc_dir}/{pdf_file}"
                body_text, body_section = read_doc_body(pdf_file)
                body = " ".join(body_text)
            title = meta['title']
            abstract = meta['abstract']
            pdf_text = " ".join([title, abstract, body])

            if len(pdf_text) > best_pdf_len:
                best_pdf_len = len(pdf_text)
                best_pdf_text = pdf_text
    return best_pdf_text


def read_doc_text(doc_metadatas, doc_dir):
    doc_pmc = read_pmc_text(doc_metadatas, doc_dir)
    if len(doc_pmc):
        return doc_pmc
    doc_pdf = read_pdf_text(doc_metadatas, doc_dir)
    return doc_pdf

def get_corduid_to_text(docs_metadata, doc_dir):
    cord_uid_to_text = {}

    for cord_uid in docs_metadata:
        corduid_metadatas = docs_metadata[cord_uid]
        corduid_text = read_doc_text(corduid_metadatas, doc_dir)
        if len(corduid_text) == 0:
            continue
        cord_uid_to_text[cord_uid] = corduid_text

    return cord_uid_to_text


"""
tokenization
"""
def text_cleaning(data):
    clean_data = re.sub(r'[^\w\s]', '', str(data).lower().strip())
    clean_data = re.sub(r' [\w\s] ', ' ', clean_data).strip()
    return clean_data

def tokenize(data):
    return word_tokenize(data)

def stemming(data):
    stems = [PorterStemmer().stem(token) for token in data]
    #stems = [Krovetz.stem(token) for token in data]
    return stems

def remove_stop_words(data, stopwords):
    tokens = [word for word in data if word not in stopwords]
    return tokens

def process_text(text, stopwords):
    clean_text = text_cleaning(text)
    text_tokens = tokenize(clean_text)
    text_tokens = remove_stop_words(text_tokens, stopwords)
    text_stems = stemming(text_tokens)
    return text_stems

def term_collection_pos(coll_vocab, doc_vocab):
    coll_term_pos = []
    for term in doc_vocab:
        coll_term_pos.append(coll_vocab[term])
    return coll_term_pos

def compute_term_idf(N, term_df):
    term_idf = term_df.astype(dtype=float)
    term_idf.data = np.log(N/term_idf.data)
    return term_idf

def generate_collection_stats(docs_metadata, text_processor, doc_dir,
                              stopwords):
    num_docs = 0
    term_doc_cnt = {}
    collection_term_cnt = {}
    vocabulary = {}
    for i, cord_uid in enumerate(docs_metadata):
        if i%100 == 0:
            print(f"Processing : {i}", end='\r', flush=True)

        """
        Reading corduid text
        """
        corduid_metadatas = docs_metadata[cord_uid]
        corduid_text = read_doc_text(corduid_metadatas, doc_dir)

        if len(corduid_text) == 0:
            continue

        corduid_tokens = text_processor(corduid_text, stopwords)

        num_docs += 1

        """
        term frequency and document count
        """
        doc_term_cnt = {}
        for term in corduid_tokens:
            index = vocabulary.setdefault(term,len(vocabulary))
            doc_term_cnt[term] = doc_term_cnt.get(term, 0) + 1

        for term, cnt in doc_term_cnt.items():
            term_doc_cnt[term] = term_doc_cnt.get(term, 0) + 1
            collection_term_cnt[term] = collection_term_cnt.get(term, 0) + cnt

    term_df = sparsemat_dict(term_doc_cnt, vocabulary)
    coll_tf = sparsemat_dict(collection_term_cnt, vocabulary)
    return coll_tf, term_df, vocabulary, num_docs

def sparsemat_dict(doc_cnt, vocabulary):
    """
    Collection and document frequency
    """
    indptr = [0]
    indices = []
    dc = []
    for term, cnt in doc_cnt.items():
        indices.append(vocabulary[term])
        dc.append(cnt)
    indptr.append(len(indices))

    dc_sparse = csr_matrix((dc, indices, indptr),
                                 shape=(1, len(vocabulary)),
                                 dtype=int)
    return dc_sparse


def get_collection_stats(collection_stat_file, collection_dir,
                         stopwords):

    if os.path.exists(collection_stat_file):
        with open(collection_stat_file, 'rb') as f:
            coll_tf, coll_t_df, coll_vocab, coll_num_doc = pickle.load(f)
    else:
        metafile = f"{collection_dir}/metadata.csv"
        docs_metadata = all_documents_metadata(metafile)

        start_time = time.time()
        coll_tf, coll_t_df, coll_vocab, coll_num_doc = generate_collection_stats(docs_metadata, process_text, collection_dir, stopwords)
        end_time = time.time()
        print(f'Time taken to obtain collection stats: {(end_time - start_time)/3600} hrs')

        with open(collection_stat_file, 'wb') as f:
            pickle.dump((coll_tf, coll_t_df, coll_vocab, coll_num_doc), f)

    return coll_tf, coll_t_df, coll_vocab, coll_num_doc


class DocVectorizer:

    def __init__(self, stopwords, vocabulary={}, doc_dir=''):
        self.doc_dir = doc_dir
        self.stopwords = stopwords

        #helper variables
        self.corduid_to_rowindex = {}

        #main variables
        self.doc_tf = None
        self.term_df = None
        self.num_docs = 0

        self.vocabulary = vocabulary
        if len(vocabulary) == 0:
            self.fixed_vocabulary = False
        else:
            self.fixed_vocabulary = True


    def compute_tf_df(self, docs_metadata, corduids, text_processor):
        self.num_docs = 0

        tf_indptr = [0]
        tf_indices = []
        tf = []

        term_doc_cnt = {}
        for i, cord_uid in enumerate(corduids):
            if i%100 == 0:
                print(f"Processing : {i}", end='\r', flush=True)

            """
            Reading corduid text
            """
            if cord_uid not in docs_metadata:
                print("ERROR: cord uid not present in metadata")
                return

            corduid_metadatas = docs_metadata[cord_uid]
            corduid_text = read_doc_text(corduid_metadatas, self.doc_dir)

            if len(corduid_text) == 0:
                continue

            corduid_tokens = text_processor(corduid_text, self.stopwords)

            if len(corduid_tokens) == 0:
                continue

            self.corduid_to_rowindex[cord_uid] = self.num_docs
            self.num_docs += 1

            """
            term frequency and document count
            """
            doc_term_cnt = {}
            for term in corduid_tokens:
                if not self.fixed_vocabulary or term in self.vocabulary:
                    index = self.vocabulary.setdefault(term, len(self.vocabulary))
                    doc_term_cnt[term] = doc_term_cnt.get(term, 0) + 1


            for term, cnt in doc_term_cnt.items():
                tf_indices.append(self.vocabulary[term])
                tf.append(cnt)
                term_doc_cnt[term] = term_doc_cnt.get(term, 0) + 1
            tf_indptr.append(len(tf_indices))


        """
        term frequency
        """
        self.doc_tf = csr_matrix((tf, tf_indices, tf_indptr),
                                     shape=(self.num_docs,
                                            len(self.vocabulary)),
                                     dtype=int)
        """
        document frequency
        """

        df_indptr = [0]
        df_indices = []
        df = []

        for term, cnt in term_doc_cnt.items():
            df_indices.append(self.vocabulary[term])
            df.append(cnt)
        df_indptr.append(len(df_indices))

        self.term_df = csr_matrix((df, df_indices, df_indptr),
                                     shape=(1, len(self.vocabulary)),
                                     dtype=int)


    def save_data(self, save_dir, tag=''):
        os.makedirs(save_dir, exist_ok=True)

        stat = (self.corduid_to_rowindex, self.doc_tf, self.term_df,
                self.num_docs, self.vocabulary)

        with open(f'{save_dir}/{tag}coll_stat.pickle', 'wb') as f:
            pickle.dump(stat, f)



    def load_data(self, save_dir, tag=''):
        filename = f'{save_dir}/{tag}coll_stat.pickle'
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                stat = pickle.load(f)
            self.corduid_to_rowindex, self.doc_tf, self.term_df, self.num_docs, self.vocabulary = stat
            return True
        return False


def text_vectorization(docs,stopwords, text_processor=None, vocabulary=None):
    num_docs = 0
    corduid_to_rowindex = {}

    if vocabulary is None:
        fixed_vocabulary = False
        vocabulary = {}
    else:
        fixed_vocabulary = True

    #vectorization
    tf_indptr = [0]
    tf_indices = []
    tf = []

    term_doc_cnt = {}
    for i, cord_uid in enumerate(docs):
        if i%1000 == 0:
            print(f"Processing : {i}", end='\r', flush=True)

        if text_processor:
            corduid_tokens = text_processor(docs[cord_uid], stopwords)
        else:
            corduid_tokens = docs[cord_uid]

        if len(corduid_tokens) == 0:
            continue

        corduid_to_rowindex[cord_uid] = num_docs
        num_docs += 1

        """
        term frequency and document count
        """
        doc_term_cnt = {}
        for term in corduid_tokens:
            if not fixed_vocabulary or term in vocabulary:
                index = vocabulary.setdefault(term, len(vocabulary))
                doc_term_cnt[term] = doc_term_cnt.get(term, 0) + 1


        for term, cnt in doc_term_cnt.items():
            tf_indices.append(vocabulary[term])
            tf.append(cnt)
            term_doc_cnt[term] = term_doc_cnt.get(term, 0) + 1
        tf_indptr.append(len(tf_indices))


    """
    term frequency
    """
    doc_tf = csr_matrix((tf, tf_indices, tf_indptr),
                                 shape=(num_docs,
                                        len(vocabulary)),
                                 dtype=int)
    """
    document frequency
    """

    df_indptr = [0]
    df_indices = []
    df = []

    for term, cnt in term_doc_cnt.items():
        df_indices.append(vocabulary[term])
        df.append(cnt)
    df_indptr.append(len(df_indices))

    term_df = csr_matrix((df, df_indices, df_indptr),
                              shape=(1, len(vocabulary)),
                              dtype=int)

    return doc_tf, term_df, vocabulary, corduid_to_rowindex


def countVectorizer(docs, vocabulary=None):
    indptr = [0]
    indices = []
    term_freq = []

    if vocabulary is None:
        fixed_vocabulary = False
        vocabulary = {}
    else:
        fixed_vocabulary = True

    for doc in docs:
        doc_term_cnt = {}

        for term in doc:
            if not fixed_vocabulary or term in vocabulary:
                index = vocabulary.setdefault(term, len(vocabulary))
                doc_term_cnt[term] = doc_term_cnt.get(term, 0) + 1

        for term, cnt in doc_term_cnt.items():
            indices.append(vocabulary[term])
            term_freq.append(cnt)
        indptr.append(len(indices))

    term_doc_matrix = csr_matrix((term_freq, indices, indptr),
                                 shape=(len(docs), len(vocabulary)),
                                 dtype=int)
    return vocabulary, term_doc_matrix


def save_ranking(save_file, ranking):
    with open(save_file, 'w') as f:
        for q_num in ranking:
            top_scores = ranking[q_num]
            corduid, score = list(zip(*top_scores.items()))
            sort_pos = np.argsort(score)[::-1]
            corduid = np.array(corduid)[sort_pos]
            score = np.array(score)[sort_pos]

            for i, (c, s) in enumerate(zip(corduid, score)):
                line = f"{q_num} Q0 {c} {i+1} {s} Suchith\n"
                f.write(line)


class RocchioReranking:

    def __init__(self, doc_tf, q_tf, coll_term_idf):
        self.doc_tf = doc_tf
        self.q_tf = q_tf
        self.term_idf = coll_term_idf

    def modify_query(self, q_vec, doc_matrix, param, nr_doc_matrix=None):
        alpha, beta, gamma = param

        if nr_doc_matrix is None:
            new_q_vec = alpha*q_vec + beta*doc_matrix.mean(axis=0)
        else:
            new_q_vec = alpha*q_vec + beta*doc_matrix.mean(axis=0) - gamma*nr_doc_matrix.mean(axis=0)
        return np.array(new_q_vec)


    def compute_doc_ltf_n(self):
        doc_ltf = self.doc_tf.astype(dtype=float)
        doc_ltf.data = np.log(doc_ltf.data)+1

        doc_n = sparse.linalg.norm(doc_ltf, axis=1)

        doc_ltf_n = doc_ltf
        doc_ltf_n.data /= doc_n.repeat(np.diff(doc_ltf_n.indptr))
        return doc_ltf_n


    def compute_query_ltf_idf_n(self):
        q_ltf = self.q_tf.astype(dtype=float)
        q_ltf.data = np.log(q_ltf.data)+1

        q_ltf_idf = q_ltf.multiply(self.term_idf)

        q_ltf_idf_n = q_ltf_idf/sparse.linalg.norm(q_ltf_idf)
        return q_ltf_idf_n


    def compute_score(self, q_vec, doc_matrix):
        return doc_matrix@q_vec.T

    def rerank_score(self, param, nr_doc_matrix=None):
        doc_ltf_n = self.compute_doc_ltf_n()
        q_ltf_idf_n = self.compute_query_ltf_idf_n()

        new_q_vec = self.modify_query(q_ltf_idf_n, doc_ltf_n,
                                        param, nr_doc_matrix)

        return self.compute_score(new_q_vec, doc_ltf_n)



class RM:

    def __init__(self, q_tf, doc_tf, coll_tf):
        self.q_tf = q_tf
        self.doc_tf = doc_tf
        self.coll_tf = coll_tf

    def compute_coll_term_prob(self):
        return self.coll_tf/self.coll_tf.sum()

    def compute_doc_length(self):
        return self.doc_tf.sum(axis=1)

    def create_uni_model(self, mu):
        coll_tp = self.compute_coll_term_prob()
        doc_len = self.compute_doc_length()

        uni_lm_dirch = (self.doc_tf + mu*coll_tp)/(doc_len + mu)
        return uni_lm_dirch


    def create_relevance_model_1(self, mu):
        uni_lm_dirch = self.create_uni_model(mu)

        log_w_by_M = np.log(uni_lm_dirch)

        log_q_by_M = self.q_tf.multiply(log_w_by_M)
        logsum_q_by_M = log_q_by_M.sum(axis=1)

        log_M = np.log(1/self.doc_tf.shape[0])

        logsum_mwq = log_M + log_w_by_M + logsum_q_by_M
        mwq = np.exp(logsum_mwq)

        wq = mwq.sum(axis=0)
        w_by_R = wq/wq.sum()
        return w_by_R, log_w_by_M


    def create_relevance_model_2(self, mu):
        uni_lm_dirch = self.create_uni_model(mu)

        M = 1/self.doc_tf.shape[0]
        w = uni_lm_dirch.sum(axis=0)*M

        M_by_w = M*uni_lm_dirch/w

        _, q_nz_pos = self.q_tf.nonzero()
        q_by_M = np.array(uni_lm_dirch[:, q_nz_pos]).T

        log_q_by_w = np.log(q_by_M@M_by_w)
        log_q_by_w = self.q_tf.data@log_q_by_w

        log_w_by_R = np.log(w) + log_q_by_w

        wq = np.exp(log_w_by_R)
        w_by_R = wq/wq.sum()

        return w_by_R, np.log(uni_lm_dirch)


    def compute_divergence(self, w_by_R, log_w_by_D):
        return log_w_by_D@w_by_R.T

    def rerank_score(self, mu, rm_type=1):
        if rm_type == 1:
            w_by_R, log_w_by_M = self.create_relevance_model_1(mu)
        elif rm_type == 2:
            w_by_R, log_w_by_M = self.create_relevance_model_2(mu)
        else:
            print("ERROR: Invalid input.")
            return

        score = np.array(self.compute_divergence(w_by_R, log_w_by_M))
        return score, w_by_R


def get_top_words(prob_w_by_R, doc_vocab, num_top_words=20):
    rev_doc_vocab = {v:k for k, v in doc_vocab.items()}
    top_words_pos = np.argsort(np.array(prob_w_by_R)[0])[::-1][:num_top_words]
    top_words = []
    for pos in top_words_pos:
        top_words.append(rev_doc_vocab[pos])
    return top_words

def save_word_expansion(save_file, top_words):
    with open(save_file, 'w') as f:
        for q_num in top_words:
            q_top_words = top_words[q_num]
            line = f"{q_num} :"
            for word in q_top_words:
                line = f"{line} {word},"
            line = f'{line[:-1]}\n'
            f.write(line)

"""
class RM2:

    def __init__(self, q_tf, doc_tf, coll_tf):
        self.q_tf = q_tf
        self.doc_tf = doc_tf
        self.coll_tf = coll_tf

    def compute_coll_term_prob(self):
        return self.coll_tf/self.coll_tf.sum()

    def compute_doc_length(self):
        return self.doc_tf.sum(axis=1)

    def create_uni_model(self, mu):
        coll_tp = self.compute_coll_term_prob()
        doc_len = self.compute_doc_length()

        uni_lm_dirch = (self.doc_tf + mu*coll_tp)/(doc_len + mu)
        return uni_lm_dirch


    def create_relevance_model(self, mu):
        uni_lm_dirch = self.create_uni_model(mu)

        M = 1/self.doc_tf.shape[0]
        w = uni_lm_dirch.sum(axis=0)*M

        M_by_w = M*uni_lm_dirch/w

        _, q_nz_pos = self.q_tf.nonzero()
        q_by_M = np.array(uni_lm_dirch[:, q_nz_pos]).T

        log_q_by_w = np.log(q_by_M@M_by_w)
        log_q_by_w = self.q_tf.data@log_q_by_w

        log_w_by_R = np.log(w) + log_q_by_w

        wq = np.exp(log_w_by_R)
        w_by_R = wq/wq.sum()

        return w_by_R, np.log(uni_lm_dirch)


    def compute_divergence(self, w_by_R, log_w_by_D):
        return log_w_by_D@w_by_R.T

    def rerank_score(self, mu):
        w_by_R, log_w_by_M = self.create_relevance_model(mu)
        return self.compute_divergence(w_by_R, log_w_by_M)
"""

