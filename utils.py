import numpy as np


class2label = {'is_a(e1,e2)': 0,
               'job_done(e1,e2)': 1, 
               'at_location(e1,e2)': 2,
               'has_a(e1,e2)': 3, 
               'has_property(e1,e2)': 4,
               'has_subevent(e1,e2)': 5, 
               'results_in(e1,e2)': 6,
               'used_for(e1,e2)': 7, 
               'work_in(e1,e2)': 8,
               'born_in(e1,e2)': 9
               }

label2class = {0: 'is_a(e1,e2)',
               1: 'job_done(e1,e2)', 
               2: 'at_location(e1,e2)',
               3: 'has_a(e1,e2)', 
               4: 'has_property(e1,e2)',
               5: 'has_subevent(e1,e2)', 
               6: 'results_in(e1,e2)',
               7: 'used_for(e1,e2)', 
               8: 'work_in(e1,e2)',
               9: 'born_in(e1,e2)'
               }


def load_glove(embedding_path, embedding_dim, vocab):
    # initial matrix with random uniform
    initW = np.random.randn(len(vocab.vocabulary_), embedding_dim).astype(np.float32) / np.sqrt(len(vocab.vocabulary_))
    # load any vectors from the word2vec
    print("Load glove file {0}".format(embedding_path))
    f = open(embedding_path, 'r', encoding='utf8')
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        idx = vocab.vocabulary_.get(word)
        if idx != 0:
            initW[idx] = embedding
    return initW
