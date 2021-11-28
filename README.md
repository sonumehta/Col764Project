# Col764Project
IR system for biomedical search:
Our goal is to build an information retrieval system for biomedical research. For the purpose of this project, we limit our analyses to one disease Covid-19 and use Covid-19 Open Research Dataset for all our analyses. Given a query related to covid-19, we want to retrieve a set of documents/research papers relevant to the query. As part of this dataset, we have access to the fullbody of the research papers. We train baseline models such as VSM, OkapiBM25, AWD-LSTM, roBERTa and Distilled roBERTa based model using the entire dataset and evaluate the performance on a set of 50 queries provided by in the Cord-19 Open Research Dataset. We then compare its performance with an Extreme Classification model (SiameseXML). SiameseXML model is trained using just the titles of the research papers and related papers as input data for an auxiliary task as described in Section 8 which generates embeddings for each token in the vocabulary. We then use these embeddings to embed both the research paper(title) and the query and use nearest neighbour search to retrieve the relevant papers for a query.
This repository contains code for various experiments that we performed on the Cord19 dataset.

1. BM25: There is one file for BM25 (BM25.py). This file  contains code to generate the data for training BM25 models, evaluating the model on a query set of 50 topics, tune the hyperparamaters and plotting a scatter plot to plot values of  metrics for all the set of hyperparamaters.

2. SiameseXML:
	2.1 SiameseXMLDataGenerator.py - This file contains code to generate the data required for training Siamese XML models. 
	This code will generate five files required for training the siameseXML model and a few other files to map the labels back to their titles etc.

	2.2 After these files are generate, clone the SiameseXML repository and follow the instructions here(https://github.com/Extreme-classification/siamesexml) to train the model. A sample config file(config.json) is present in this repository.

	2.3 Once the model is trained, Z.pkl file is generated which are the embeddings of the tokens. Use that file and run SiameseXMLEvaluator.py to evaluate the model on query dataset of 50 topics.
