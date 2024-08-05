'''Trains and evaluates each model'''
import decisiontrees_466 as dt
import knn_466 as knn
import fnn_466 as fnn
import df_code
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import torch
import math

# Dataset selection
# *** make sure to pick which dataset within df_code.py ***
#dataset = 'all'
dataset = 'short'
X_train, X_test, y_train, y_test, class_names = df_code.main(dataset)


def run():
    if dataset == 'all':
        print('The entire dataset:')
    else:
        print('The edited dataset:')
    print()

    print('Training Feedforward Neural Network...')
    fnn_instance = fnn.fnn(dataset)
    fnn_model = fnn.main(X_train, X_test, y_train, fnn_instance)
    print('Done Feedforward Neural Network')

    print('Training Decision Trees...')
    dt_model = dt.main(X_train, X_test, y_train)
    print('Done Decision Trees')

    print('Training K-Nearest Neighbor...')
    knn_model = knn.main(X_train, X_test, y_train)
    print('Done K-Nearest Neighbor')
   
    # Baselines
    classes_counts = Counter(list(y_train))
    topclass = classes_counts.most_common()[0] 
    majority_class_classifier = Counter(list(y_test))[topclass[0]] / len(y_test)
    
    k = math.ceil(0.05*len(class_names))
    topkclass = classes_counts.most_common()[0:k]
    top_k_correct = sum(1 for label in y_test if label in [x[0] for x in topkclass])
    top_k_classifier = top_k_correct / len(y_test)
    
    random_chance = 1/len(class_names)
    top_k_rand = math.ceil(0.05*len(class_names))/len(class_names)
    
    # Dictionaries to store accuracies
    accuracies = {'Comparisons': [random_chance, majority_class_classifier]}
    top_k ={'Comparisons': [top_k_rand, top_k_classifier]}

    # Evaluate
    for model in [ ('Feedforward Neural Network',fnn_model),('Decision Trees',dt_model), ('K-Nearest Neighbors',knn_model)]:
        print("Getting training accuracies...")
        acc, k_pred = get_evals(model,X_train, y_train, k)
        accuracies[model[0]] = [acc]
        top_k[model[0]] = [k_pred]

        print("Getting test accuracies...")
        acc, k_pred = get_evals(model,X_test, y_test, k)
        accuracies[model[0]].append(acc)
        top_k[model[0]].append(k_pred) 

    for key, value in accuracies.items():
        print(key)
        print("Accuracy:", value[0])
        print()

    
    # For graphs
    print("accuracies.append(",accuracies,")")
    print("top_k_acc.append(", top_k,")")


def get_evals(model, feats, obs, top_k):
    '''Returns accuracies'''
    if model[0] == 'Feedforward Neural Network':
        label_encoder = model[1][1]
        preds = model[1][0].eval(feats, label_encoder)
        top_k_preds = model[1][0].eval_top_k_probs(feats, label_encoder, top_k=top_k)
    else:
        preds = model[1].predict(feats)
        top_k_probs = np.argsort(model[1].predict_proba(feats), axis=1)[:, -top_k:]
        top_k_preds = [model[1].classes_[top_k_probs[i]] for i in range(len(top_k_probs))]
    
    accuracy = accuracy_score(obs, preds)
    top_k_acc = k_pred_acc(obs, top_k_preds)
    #print(top_k_preds)
    #input()
    return accuracy, top_k_acc

def k_pred_acc(y_test, top_k_preds):
    '''Converts the k predictions into an accuracy'''
    count = 0
    for true_class, preds in zip(y_test, top_k_preds):
        count +=  sum(1 for pred in preds if pred == true_class)
    return count/len(y_test)


run()
