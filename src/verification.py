#!/usr/bin/env python
# -*- coding: utf-8 -*-

import glob
import os
import re

from operator import itemgetter
from collections import Counter, defaultdict
from itertools import product

import numpy as np
np.random.seed(1337654)
rnd = np.random.RandomState(1337987)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd
from scipy.stats import ks_2samp

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.cross_validation import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, average_precision_score

from nltk import word_tokenize

from mpl_toolkits.axes_grid.anchored_artists import AnchoredText


linebreak = re.compile(r'\-\s*\n\s*')
double_quotes = re.compile(r'“|”')
whitespace = re.compile(r'\s+')
ae = re.compile(r'æ')

from numba import jit

@jit
def minmax(x, y, rnd_feature_idxs):
    mins, maxs = 0.0, 0.0

    for i in rnd_feature_idxs:
        a, b = x[i], y[i]

        if a >= b:
            maxs += a
            mins += b
        else:
            maxs += b
            mins += a

    return 1.0 - (mins / (maxs + 1e-6)) # avoid zero division

def simple_stats(documents):
    # test length distribution:
    lenghts = [len(d) for d in documents]
    sns.distplot(lenghts)
    plt.title('Documents lengths')
    plt.savefig('../figures/doc_lengths.png')
    print('# docs:', len(lenghts))
    print('mu doc length:', np.mean(lenghts))
    print('sigma doc length:', np.std(lenghts))


def words_per_author(authors, documents):
    cnt = Counter()
    for a, d in zip(authors, documents):
        cnt[a] += len(d)
    items = cnt.most_common(15)[::-1]
    names, cnts = zip(*items)
    pos = np.arange(len(names))
    plt.clf()
    plt.barh(pos, cnts, color='blue')
    plt.yticks(pos + 0.5, names)
    plt.tight_layout()
    plt.savefig('../figures/words_per_author.pdf')

def texts_per_author(authors, documents):
    cnt = Counter()
    for a, d in zip(authors, documents):
        cnt[a] += 1
    items = cnt.most_common(15)[::-1]
    names, cnts = zip(*items)
    pos = np.arange(len(names))
    plt.clf()
    plt.barh(pos, cnts, color='blue')
    plt.yticks(pos + 0.5, names)
    plt.tight_layout()
    plt.savefig('../figures/texts_per_author.pdf')


def plot_confusion_matrix(cm, target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    new_style = {'grid': False}
    matplotlib.rc('axes', **new_style)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.tick_params(labelsize=6)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=90)
    plt.yticks(tick_marks, target_names)
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def tokens_from_file(filename, text_cutoff=None):
    text = open(filename).read().strip().lower()

    # regex cleanup:
    text = linebreak.sub('', text)
    text = double_quotes.sub('"', text)
    text = text.replace('—', ' - ')
    text = ae.sub('ae', text)
    text = whitespace.sub(' ', text)

    tokens = word_tokenize(text)
    tokens = [t.strip() for t in tokens if t.strip()]

    if text_cutoff:
        tokens = tokens[:text_cutoff]

    return tokens

def load_data(text_cutoff=None, segment_size=2000, max_nb_segments=None, minimal_text_size=None):
    instances = []
    for filename in sorted(glob.glob('../data/*.txt')):
        bn = os.path.basename(filename).replace('.txt', '').lower()
        print(bn)
        
        author, title, genre = bn.split('_')

        if genre == 'book':
            continue

        title = title.lower().replace(' ', '-')
        title = ''.join([c for c in title if c.isalpha() or c == '-'])
        if len(title) > 50:
            title = title[:50] + '[...]'
        
        tokens = tokens_from_file(filename, text_cutoff)

        if minimal_text_size and len(tokens) < minimal_text_size:
            continue

        if len(tokens) < segment_size:
            instances.append([author, title, tokens])
        else:
            start_idx, end_idx = 0, segment_size
            cnt = 1
            while end_idx <= len(tokens):
                instances.append([author, title, tokens[start_idx : end_idx]])
                start_idx += segment_size
                end_idx += segment_size
                cnt += 1

                if max_nb_segments:
                    if cnt >= max_nb_segments:
                        break

    return zip(*instances)


def main():
    max_nb_segments = None
    minimal_text_size = 1000
    text_cutoff = None
    full_nb_features = 50000
    nb_imposters = 30

    # find out length of target text:
    segment_size = len(tokens_from_file('../data/AnonA_Observations on the nature and importance of geology_article.txt', None))
    print('segment size = size(anon_a) ->', segment_size)

    authors, titles, documents = load_data(segment_size=segment_size,
                                           text_cutoff=text_cutoff,
                                           max_nb_segments=max_nb_segments,
                                           minimal_text_size=minimal_text_size)

    simple_stats(documents)
    words_per_author(authors, documents)
    texts_per_author(authors, documents)

    documents = [' '.join(d) for d in documents]

    word_vectorizer = TfidfVectorizer(max_features=full_nb_features, analyzer='word',
                                 ngram_range=(1, 1), use_idf=True, token_pattern=r"\b\w+\b")
    X_word = word_vectorizer.fit_transform(documents).toarray()

    ngram_vectorizer = TfidfVectorizer(max_features=full_nb_features, analyzer='char_wb',
                                 ngram_range=(4, 4), use_idf=True, token_pattern=r"\b\w+\b")
    X_ngram = ngram_vectorizer.fit_transform(documents).toarray()

    X = np.hstack((X_word, X_ngram))
    feature_names = [w + '(w)' for w in word_vectorizer.get_feature_names()] + \
                        [ng + '(ng)' for ng in ngram_vectorizer.get_feature_names()]

    # unit norm scaling:
    X = normalize(X, norm='l2')

    df = pd.DataFrame(X, columns=feature_names)
    
    df.insert(0, 'title_', titles)
    df.insert(0, 'author_', authors)

    #candidate_authors = sorted(('jameson', 'boue', 'grant', 'weaver', 'fleming', 'lyell', 'cheek'))
    candidate_authors = sorted(('jameson', 'boue', 'grant', 'weaver', 'fleming', 'lyell'))

    columns = ['author_', 'title_'] + candidate_authors

    results = pd.DataFrame(columns=columns)

    for row in df.itertuples(index=False):
        test_author, test_title = row[0], row[1]
        test_vector = np.array(row[2:])

        if not (test_author in candidate_authors or test_author.startswith('anon')):
            continue

        curr_results = [test_author, test_title]
        for candidate_author in candidate_authors:
            target_vectors = df[(df['author_'] == candidate_author) & (df['title_'] != test_title)].as_matrix(feature_names)
            imposter_vectors = df[(df['author_'] != candidate_author)].as_matrix(feature_names)
            
            sigmas = []
            for iteration in range(1000):

                # rnd feature indices
                rnd_feature_idxs = list(range(full_nb_features))
                rnd.shuffle(rnd_feature_idxs)
                rnd_feature_idxs = rnd_feature_idxs[:int(full_nb_features / 2.)]

                # rnd imposter indices:
                rnd_imp_idxs = list(range(imposter_vectors.shape[0]))
                rnd.shuffle(rnd_imp_idxs)
                rnd_imp_idxs = rnd_imp_idxs[:nb_imposters]

                # minimal target and non-target distance:
                min_target = np.min([minmax(test_vector, v, rnd_feature_idxs=rnd_feature_idxs) for v in target_vectors])
                max_target = np.min([minmax(test_vector, imposter_vectors[g, :], rnd_feature_idxs=rnd_feature_idxs) for g in rnd_imp_idxs])

                if min_target < max_target:
                    sigmas.append(1)
                else:
                    sigmas.append(0)

            score = sum(sigmas, 0.0) / len(sigmas)
            curr_results.append(score)

        results.loc[len(results)] = curr_results
        print(results)

    # compute naive attribution accuracies:
    true_authors, predicted_authors = [], []
    for row in results.itertuples(index=False):
        # actual author:
        true_author = row[0]
        if true_author.startswith('anon'):
            continue

        true_authors.append(true_author)

        # predicted author:
        scores = row[2:]
        top_idx = np.argmax(scores)
        predicted_author = candidate_authors[top_idx]
        predicted_authors.append(predicted_author)

    print('naive attribution accuracy:', accuracy_score(true_authors, predicted_authors))

    # plot 
    plt.clf()
    T = true_authors
    P = predicted_authors
    cm = confusion_matrix(T, P, labels=candidate_authors)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=2)
    sns.plt.figure()
    plot_confusion_matrix(cm_normalized, target_names=candidate_authors)
    sns.plt.savefig('../figures/naive_attrib_conf_matrix.pdf')

    # collect results for precision and recall:
    gold, silver = [], []
    for row in results.itertuples(index=False):
        true_author = row[0]
        scores = row[2:]
        if true_author.startswith('anon'):
            continue
        else:
            for cand_author, score in zip(candidate_authors, scores):
                silver.append(score)
                if cand_author == true_author:
                    gold.append(1.0)
                else:
                    gold.append(0.0)

    precisions, recalls, thresholds = precision_recall_curve(gold, silver)
    F1s = 2 * (precisions * recalls) / (precisions + recalls)
    best_f1_idx = np.argmax(F1s)
    best_f1 = F1s[best_f1_idx]
    best_threshold = thresholds[best_f1_idx]
    best_prec, best_rec = precisions[best_f1_idx], recalls[best_f1_idx]
    
    plt.clf()
    fig, ax = sns.plt.subplots()
    plt.plot(recalls, precisions, color='navy', label='all features')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.scatter(best_rec, best_prec, c='black')

    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')

    at = AnchoredText('Optimal F1: '+str(format(best_f1, '.3f'))+'\nFor theta = '+str(format(best_threshold, '.3f')), prop=dict(size=10), frameon=True, loc=1)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    plt.legend(loc="lower left")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig('../figures/prec_rec.pdf')

    results = results.set_index('author_')
    results.to_csv('../figures/results.csv')

    # plot heatmap:
    X = results.as_matrix(candidate_authors)
    fig, ax = plt.subplots(figsize=(7,12))
    heatmap = ax.pcolor(X, cmap=matplotlib.cm.Blues)
    ax.set_xticks(np.arange(X.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(X.shape[0])+0.5, minor=False)
    ax.set_xticklabels(candidate_authors, minor=False, rotation=90, size=8)
    ax.set_yticklabels(results.index, minor=False, size=5)
    plt.ylim(0, len(results))
    plt.tight_layout()
    plt.savefig('../figures/verif_heatmap.pdf')
    plt.clf()

    # plot distributions using kdeplot:
    sns.set_style("dark")
    ax.set_xlim([-1, 1])
    
    diff_auth_pairs, same_auth_pairs = [], [] 
    for g, s in zip(gold, silver):
        if g == 1.0:
            same_auth_pairs.append(s)
        else:
            diff_auth_pairs.append(s)

    diff_auth_pairs = np.asarray(diff_auth_pairs, dtype='float64')
    same_auth_pairs = np.asarray(same_auth_pairs, dtype='float64')

    fig, ax = sns.plt.subplots()
    c1, c2 = sns.color_palette('Set1')[:2]
    sns.plt.xlim(0, 1)

    sns.kdeplot(diff_auth_pairs, shade=True, legend=False, c=c1, lw=0.5, label='diff. author pairs', ax=ax)
    sns.kdeplot(same_auth_pairs, shade=True, legend=False, c=c2, lw=0.5, label="same author pairs",  ax=ax)

    sns.plt.legend(loc=0)

    # annotate plot:
    # test for signifiance via Kolmogorov-Smirnov:
    D, p = ks_2samp(diff_auth_pairs, same_auth_pairs)
    print("\t\t\t- KS: D = %s (p = %s)" %(D, p))

    ax.xaxis.set_major_formatter(sns.plt.NullFormatter())
    ax.yaxis.set_major_formatter(sns.plt.NullFormatter())

    if p < 0.001:
        at = AnchoredText("KS: "+str(format(D, '.3f')+'\np < 0.001'), prop=dict(size=12), frameon=True, loc=2)
    else:
        at = AnchoredText("KS: "+str(format(D, '.3f')+'\np > 0.001'), prop=dict(size=12), frameon=True, loc=2)

    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    sns.axes_style()

    sns.plt.savefig('../figures/distr.pdf')
    sns.plt.clf()    


if __name__ == '__main__':
    main()