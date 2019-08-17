# Copyright 2017 Benjamin Riedel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Import relevant packages and modules
from util import *
import random
import tensorflow as tf
import time
from randomList import *
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
import warnings
import argparse


# print accuracy
def calculateAcc(test_pred, correct_pred):
    error_count = 0

    for i in range(0, len(correct_pred)):
        if test_pred[i] != correct_pred[i]:
            error_count = error_count + 1
    
    print("Error: " + str(error_count))
    acc = (len(correct_pred) - error_count) / len(correct_pred)
    return acc

def calculatePredicAccuracy(sess, test_set, test_stances,features_pl, keep_prob_pl,predict):
    # random_set, random_set_stance = generate_random_set_stance(train_set, train_stances, 950)
    # Predict
    warnings.filterwarnings('always') 
    test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
    test_pred = sess.run(predict, feed_dict=test_feed_dict)

    correct_pred = test_stances


    accuracy = accuracy_score(correct_pred, test_pred)
    f1Score = f1_score(correct_pred, test_pred, average='macro',labels=np.unique(test_pred)) 

    print('Test accuracy:' , accuracy)
    print('Test F1 score: ======',f1Score)

    print('an epoch done >>>')
    return accuracy




def process(args):
    # Set file names
    file_train_instances = "train_stance_done.csv"
    file_train_bodies = "train_bodies_done.csv"
    file_test_instances = "test_stances_unlabeled.csv"
    file_test_bodies = "test_bodies.csv"
    file_predictions = 'predictions_test.csv'
    # Initialise hyperparameters
    r = random.Random()
    lim_unigram = 5000
    target_size = 4
    hidden_size = 100
    train_keep_prob = 0.6
    l2_alpha = 0.00001
    learn_rate = 0.01
    clip_ratio = 5
    batch_size_train = 1
    epochs = 50
    # Load data sets
    raw_train = FNCData(file_train_instances, file_train_bodies)
    raw_test = FNCData(file_test_instances, file_test_bodies)
    n_train = len(raw_train.instances)
    print('n_train: ========>>>>>', n_train)
    # Process data sets
    train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
        pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)
    feature_size = len(train_set[0])
    test_set = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)
    # Create placeholders
    features_pl = tf.placeholder(tf.float32, [None, feature_size], 'features')
    stances_pl = tf.placeholder(tf.int64, [None], 'stances')
    keep_prob_pl = tf.placeholder(tf.float32)
    batch_size = tf.shape(features_pl)[0]
    # Define multi-layer perceptron
    hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(features_pl, hidden_size)), keep_prob=keep_prob_pl)
    logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, target_size), keep_prob=keep_prob_pl)
    logits = tf.reshape(logits_flat, [batch_size, target_size])
    # Define L2 loss
    tf_vars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha
    # Define overall loss
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, stances_pl) + l2_loss)
    # Define prediction
    softmaxed_logits = tf.nn.softmax(logits)
    print('softmaxed_logits: ', softmaxed_logits)
    predict = tf.arg_max(softmaxed_logits, 1)
    # predict = softmaxed_logits
    print('predict: ', predict )
    # Define optimiser
    opt_func = tf.train.AdamOptimizer(learn_rate)
    # tf.clip_by_global_norm 梯度剪裁  t_list,  ＃常輸入梯度 clip_norm, #裁剪率
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tf_vars), clip_ratio)
    opt_op = opt_func.apply_gradients(zip(grads, tf_vars))
    trainData, trainStances,testData, testStances = splitTrainTest(train_set, train_stances, 0.8)
    # Add ops to save and restore all the variables.


    if args.model == 'load':
        with tf.Session() as sess:
            load_model2(sess,args.model_save_path)
            print('load done')
            # Predict
            test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
            test_pred = sess.run(predict, feed_dict=test_feed_dict)

            save_predictions(test_pred, file_predictions)

    elif args.model == 'train':



        saver = tf.train.Saver()
        # Perform training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            accs = []
            # side effect free function
            # random_set, random_set_stance = generate_random_set_stance(train_set, train_stances, 3792)
            for epoch in range(epochs):
                print('epoch number:-----------------------------',epoch)
                total_loss = 0
                # indices = list(range(n_train))
                indices = list(range(len(trainData)))
                r.shuffle(indices)

                iteration_size = len(trainData) // batch_size_train
                for i in range(iteration_size):
                    batch_indices = indices[i * batch_size_train: (i + 1) * batch_size_train]
                    batch_features = [trainData[i] for i in batch_indices]
                    batch_stances = [trainStances[i] for i in batch_indices]
                    batch_feed_dict = {features_pl: batch_features, stances_pl: batch_stances, keep_prob_pl: train_keep_prob}
                    _, current_loss = sess.run([opt_op, loss], feed_dict=batch_feed_dict)
                    total_loss += current_loss
                    # print('current loss == ',current_loss)
                acc = calculatePredicAccuracy(sess, testData, testStances,features_pl, keep_prob_pl,predict)
                accs.append(acc)
                average_loss = total_loss *1. / iteration_size
                print('average loss:', average_loss)
                save_path = saver.save(sess, args.model_save_path)
                # print('save done')
       
            total_acc = sum(accs) / epochs
            print("average acc: " + str(total_acc)) 

# factory method

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_save_path', default='tmp_model/model.ckpt')
    parser.add_argument('--model', default='train')
    return parser

def displayFunctionality():
    print('1. training a model.')
    print('2. load model from checkpoint.')

def getArgsModelType(choice):
    if choice == '1':
        return 'train'
    elif choice == '2':
        return 'load'

if __name__ == "__main__":

    # Prompt for mode
    parser = createParser()
    args = parser.parse_args()
    print('save model place: ',args.model_save_path)

    displayFunctionality()

    choice = input('enter choice: ')
    args.model = getArgsModelType(choice)

    print(args.model)
    starttime = time.time()
    process(args)
    endtime = time.time()
    print(endtime - starttime,' seconds')    



# Save predictions
#save_predictions(test_pred, file_predictions)





