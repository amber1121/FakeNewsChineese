from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

if __name__ == '__main__':

    from sklearn.feature_extraction.text import TfidfVectorizer
    corpus = [
         'This is the first document.',
         'This document is the second document.',
         'And this is the third one.',
         'Is this the first document?',
    ]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    print(X)
    print(vectorizer.get_feature_names())

    id_ref = {}
    heads = ['I am a girl', 'I wanna be a bitch']
    body_ids = [1, 2]
    bodies = ['fuck you bitch', 'i wannna fuck you']
    # heads = heads.split(' ')
    # body_ids = body_ids.split(' ')
    head = 'girl'
    body_id = 'love'


    for i, elem in enumerate(heads + body_ids):
        id_ref[elem] = i
    # print(id_ref)
    # Create vectorizers and BOW and TF arrays for train set
    bow_vectorizer = CountVectorizer(max_features=3)
    bow = bow_vectorizer.fit_transform(heads + bodies)# Train set only
    print(bow_vectorizer.get_feature_names())
    # # print(bow.toarray())
    tfreq_vectorizer = TfidfTransformer(use_idf=False).fit(bow)
    tfreq = tfreq_vectorizer.transform(bow).toarray()  # Train set only
    print('tfreq : ',tfreq)
    print('tfreq[id_ref[I am a girl]]',tfreq[id_ref['I am a girl']])

    print('reshape=====',tfreq[id_ref[1]].reshape(1, -1))
    # head_tf = tfreq[id_ref[head]].reshape(1, -1)
    # body_tf = tfreq[id_ref[body_id]].reshape(1, -1)
    #
    # feat_vec = np.squeeze(np.c_[head_tf, body_tf])
    # print(feat_vec)