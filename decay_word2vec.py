import numpy as np
import gensim
from gensim import utils
from random import shuffle
import multiprocessing
import logging


model_name = "AP_word2vec_decay_300"


epochs = 10
alpha = 0.04
size = len(data)
factor = 1
train = data[:size/factor]


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# removing iter as that we implement using delay epochs
# find the use of hs and sample


model = gensim.models.Word2Vec(
    sentences=None,
    size=300,
    alpha=alpha, 
    window=10,
    min_count=2,
    seed=6,
    workers=multiprocessing.cpu_count() - 2, 
    min_alpha=alpha, 
    sg=1,
    sample = None,
    hs = 1)

model.build_vocab(train)
model.intersect_word2vec_format(data_path + 'google_vectors.bin.gz', binary = True)


for epoch in range(epochs):
	shuffle(train)
	print '\nEpoch: %d'%(epoch + 1)
	model.train(train)
    model.save_word2vec_format(data_path + model_directory + 'word_vectors_'+ model_name, binary = True)
    model.save(data_path + model_directory + 'model_'+ model_name)
	model.alpha = model.alpha / (1 + epoch)
	model.min_alpha = model.alpha


# model.init_sims(replace=True)
model.save_word2vec_format(data_path + model_directory + 'word_vectors_'+ model_name, binary = True)
model.save(data_path + model_directory + 'model_'+ model_name)

gc.collect()