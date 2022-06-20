import re
import os
import sys
import time
import copy
import math
import argparse
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtools.optim import RangerLars
import gensim.downloader
import itertools

from scipy.stats import ortho_group
from scipy.optimize import linear_sum_assignment as linear_assignment
import umap
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

import numpy as np
from tqdm import tqdm_notebook
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from utils import AverageMeter
from collections import OrderedDict

import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from gensim.corpora.dictionary import Dictionary
from pytorch_transformers import *
from sklearn.mixture import GaussianMixture
import scipy.stats
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS

from gensim.models.coherencemodel import CoherenceModel
from tqdm import tqdm
import scipy.sparse as sp
import nltk

from data import *
import warnings
warnings.filterwarnings("ignore")


class CTFIDFVectorizer(TfidfTransformer):
    def __init__(self, *args, **kwargs):
        super(CTFIDFVectorizer, self).__init__(*args, **kwargs)

    def fit(self, X: sp.csr_matrix, n_samples: int):
        """Learn the idf vector (global term weights) """
        _, n_features = X.shape
        df = np.squeeze(np.asarray(X.sum(axis=0)))
        idf = np.log(n_samples / df)
        self._idf_diag = sp.diags(idf, offsets=0,
                                  shape=(n_features, n_features),
                                  format='csr',
                                  dtype=np.float64)
        return self

    def transform(self, X: sp.csr_matrix) -> sp.csr_matrix:
        """Transform a count-based matrix to c-TF-IDF """
        X = X * self._idf_diag
        X = normalize(X, axis=1, norm='l1', copy=False)
        return X
    
def coherence_normalize(doc):
    return [word for word in doc.lower().split()]

def get_internal_coherence(topics, texts, coherence='c_npmi', topn=10):
    gensim_dict = Dictionary(texts)
    model = CoherenceModel(topics=topics, texts=texts, dictionary=gensim_dict, coherence=coherence, topn=topn)
    coherence_val = model.get_coherence()
    print(coherence, ':', coherence_val)
    return coherence_val


def vect2gensim(vectorizer, dtmatrix):
     # transform sparse matrix into gensim corpus and dictionary
    corpus_vect_gensim = gensim.matutils.Sparse2Corpus(dtmatrix, documents_columns=False)
    dictionary = Dictionary.from_corpus(corpus_vect_gensim,
        id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))

    return (corpus_vect_gensim, dictionary)

class BertWordFromTextEncoder:
    def __init__(self, valid_vocab=None, device=None):
        self.device = device
        self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.model, self.bert_tokenizer = self.load_bert_models()
        self.w2vb = {} #embeds_sum
        self.w2vc = {} #counts
        self.compounds = set()
        self.agg_by = ""
        self.use_full_vocab = False

        if valid_vocab is None:
            print("Provide list of vocab words.")
            sys.exit(1)

        elif valid_vocab == -1:
            self.use_full_vocab = True
            print("Extract embeddings with full vocab")

        else:
            print(f"Extract embeddings with restricted vocab, {len(valid_vocab)} words")

        self.valid_vocab = valid_vocab

    def test_encoder(self):

        input_ids = torch.tensor([self.bert_tokenizer.encode('Here is some text to \
            encode')]).to(self.device)
        last_hidden_states = self.model(input_ids)[0][0]
        print("Bert models are working fine\n")


    def load_bert_models(self):
        model_class = BertModel
        tokenizer_class = BertTokenizer
        pretrained_weights = 'bert-base-uncased'
        tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        model = model_class.from_pretrained(pretrained_weights,
                output_hidden_states=True).to(self.device)

        return model, tokenizer
    
    def _add_word(self, compound_word, compound_ixs, embeds):

        word = "".join(compound_word).lower()

        if self.agg_by=="firstword":
            w = compound_ixs[0] 
            emb = embeds[w]
        elif self.agg_by=="average":
            total_emb = 0
            for w in compound_ixs:
                total_emb += embeds[w]
            emb = total_emb/len(compound_ixs)

        emb = emb.cpu().detach().numpy()

        if self.use_full_vocab:
            pass
        else:
            if word not in self.valid_vocab:
                return

        if len(compound_ixs)>1:
            self.compounds.add(word)

        if word in self.w2vb:
            self.w2vb[word] += emb
            self.w2vc[word] += 1
        else:
            self.w2vb[word] = emb
            self.w2vc[word] = 1

    def eb_dump(self, save_fn):
        print("saving embeddings")

        all_vecs = []
        for word in self.w2vb:
            #word = word.encode('utf-8', 'ignore').decode('utf-8')

            mean_vector = np.around(self.w2vb[word]/self.w2vc[word], 8)
            vect = np.append(word, mean_vector)
            all_vecs.append(vect)

        #np.savetxt(f'embeds/bert_embeddings{i}-layer{args.layer}.txt', all_vecs, fmt = '%s', delimiter=" ")

        np.savetxt(save_fn, np.vstack(all_vecs), fmt = '%s', delimiter=" ", encoding='utf-8')
        print(f"{len(all_vecs)} vectors saved to {save_fn}")
        print(f"{len(self.compounds)} compound words saved to: compound_words.txt")
        
        with open('compound_words.txt', 'w') as f:
            f.write("\n".join(list(self.compounds)))

        sys.stdout.flush()

    def encode_docs(self, docs=[], agg_by="firstword", save_fn="", layer=12):
        self.agg_by = agg_by

        if len(save_fn)==0:
            save_fn = f"{args.data}-bert-layer{args.nlayer}-{agg_by}.txt"
            print(f"No save filename provided, saving to: {save_fn}")

        start = time.time()
        with torch.no_grad(): 
            for i, doc in enumerate(docs):
                if i%(int(len(docs)/100))==0:
                    timetaken = np.round(time.time() - start, 1)
                    print(f"{i+1}/{len(docs)} done, elapsed(s): {timetaken}")
                    sys.stdout.flush()
                
                # Assume a sentence as the window for contextualised embeddings.
                sents = self.sent_tokenizer.tokenize(doc)

                for sent in sents:
                    words = self.bert_tokenizer.tokenize(sent)

                    if len(words) > 500:
                        new_sents = [""]
                        fragment = ""
                        nsubwords = 0
                        currentlength = 0
                        for w in words:
                            nsubwords += 1
                            if w.startswith("##"):
                                fragment += w.replace("##", "")
                                currentlength += 1
                            else:
                                if nsubwords > 500:
                                    new_sents.append("")
                                new_sents[-1] += " " + fragment
                                fragment = w
                                currentlength = 1
                        new_sents[-1] += " " + fragment
                        new_sents = [s[1:] for s in new_sents]

                        detok = " ".join(new_sents)
                        if not words == self.bert_tokenizer.tokenize(detok):
                            pdb.set_trace()
                    else:
                        new_sents = [sent]

                    for sent in new_sents:
                        if len(new_sents) > 1:
                            words = self.bert_tokenizer.tokenize(sent)
                        try:
                            input_ids = torch.tensor([self.bert_tokenizer.encode(sent)]).to(self.device)
                        except:
                            pdb.set_trace()
                        # words correspond to input_ids correspond to embeds

                        try:
                            embeds = self.model(input_ids)[-2:][1][layer][0]
                        except Exception as e:
                            print(f"Crashed during encoding sentence: {sent}\n\n")
                            print(f"Error message:", e)
                            sys.exit(1)

                        compound_word = []
                        compound_ixs = []
                        full_word = ""

                        for w, word in enumerate(words):


                            if word.startswith('##'):
                                compound_word.append(word.replace('##',''))
                                compound_ixs.append(w)


                            else:
                                # add the previous word
                                # reset the compound word
                                if w!=0:
                                    try:
                                        self._add_word(compound_word, compound_ixs, embeds)
                                    except:
                                        pdb.set_trace()

                                compound_word = [word]
                                compound_ixs = [w]

                            if w == len(words)-1:
                                try:
                                    self._add_word(compound_word, compound_ixs, embeds)
                                except:
                                    pdb.set_trace()

        self.eb_dump(save_fn)
        
def create_vocab_preprocess(stopwords, data, vocab, preprocess, process_data=False):
    word_to_file = {}
    word_to_file_mult = {}
    strip_punct = str.maketrans("", "", string.punctuation)
    strip_digit = str.maketrans("", "", string.digits)

    process_files = []
    for file_num in range(0, len(data)):
        words = data[file_num].lower().translate(strip_punct).translate(strip_digit)
        words = words.split()
        #words = [w.strip() for w in words]
        proc_file = []

        for word in words:
            if word in stopwords or (word not in vocab and len(vocab)) or word =="dlrs" or word == "revs":
                continue
            if word in word_to_file:
                word_to_file[word].add(file_num)
                word_to_file_mult[word].append(file_num)
            else:
                word_to_file[word]= set()
                word_to_file_mult[word] = []

                word_to_file[word].add(file_num)
                word_to_file_mult[word].append(file_num)

        process_files.append(proc_file)

    for word in list(word_to_file):
        if len(word_to_file[word]) <= preprocess  or len(word) <= 3:
            word_to_file.pop(word, None)
            word_to_file_mult.pop(word, None)

    print("Files:" + str(len(data)))
    print("Vocab: " + str(len(word_to_file)))
    return word_to_file, word_to_file_mult, data


def vect2gensim(vectorizer, dtmatrix):
     # transform sparse matrix into gensim corpus and dictionary
    corpus_vect_gensim = gensim.matutils.Sparse2Corpus(dtmatrix, documents_columns=False)
    dictionary = Dictionary.from_corpus(corpus_vect_gensim,
        id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))

    return (corpus_vect_gensim, dictionary)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text

def read_entity_file(file, id_to_word, vocab):
    data = []
    word_index = {}
    index = 0
    mapping = None
    if id_to_word != None:
        mapping = create_id_dict(id_to_word)

    for line in open(file):
        embedding = line.split()
        if id_to_word != None:
            embedding[0] = mapping[embedding[0]][1:]
        if embedding[0] in vocab:
            word_index[embedding[0]] = index
            index +=1
            embedding = list(map(float, embedding[1:]))
            data.append(embedding)

    print("KG: " + str(len(data)))
    return data, word_index

def find_intersect_unique(word_index, vocab, data):
    words = []
    vocab_embeddings = []

    intersection = set(word_index.keys()) & set(vocab.keys())
    print("Intersection: " + str(len(intersection)))

    intersection = np.sort(np.array(list(intersection)))
    for word in intersection:
        vocab_embeddings.append(data[word_index[word]])
        words.append(word)

    vocab_embeddings = np.array(vocab_embeddings)

    return vocab_embeddings, words

def cluster(intersection, words_index_intersect, num_topics, rerank, weights, topics_file, rand):
    labels, top_k = GMM_model(intersection, words_index_intersect, num_topics, rerank, rand)
    bins, top_k_words = sort(labels, top_k,  words_index_intersect)
    return top_k_words, np.array(top_k)

def find_top_k_words(k, top_vals, vocab):
    ind = []
    unique = set()
    for i in top_vals:
        word = vocab[i]
        if word not in unique:
            ind.append(i)
            unique.add(vocab[i])
            if len(unique) == k:
                break
    return ind