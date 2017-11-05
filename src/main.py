# (c) Catalin-Stefan Tiseanu, 2015-

import os
import parmap
import numpy as np
import joblib
import subprocess
import argparse

from tqdm import tqdm
import pandas as pd

from collections import defaultdict
from collections import Counter

import pickle
import sys

import time

from sklearn import linear_model
import logging

NUM_LANGUAGES = 176

NUM_MIXTURES = 2048
NUM_FEATURES = 62

MODEL_NAME = "2048_62_emm"

logging.root.setLevel(level=logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')

# Computes helpful mappings, such as language->id mapping
def compute_general_dict(train_truth_filename, test_filename):
    train_truth = pd.read_csv(train_truth_filename)
    test_list = open(test_filename).read().splitlines()

    # Compute language to file index
    language_to_id = {}
    id_to_language = {}
    language_id = 0
    language_to_file = defaultdict(list)
    labels = []

    for index, entry in tqdm(train_truth.iterrows()):
        language_to_file[entry.language].append(entry.filename)
        if entry.language not in language_to_id:
            language_to_id[entry.language] = language_id
            id_to_language[language_id] = entry.language
            language_id += 1
        labels.append(str(language_to_id[entry.language]))

    pickle.dump(language_to_file, open("data/language_to_file", "wb"))
    pickle.dump(language_to_id, open("data/language_to_id", "wb"))
    pickle.dump(id_to_language, open("data/id_to_language", "wb"))
    open("data/train_labels", "w").write("\n".join(labels))

    logging.info('Computed auxiliary mappings, saved to data/')


# Takes in a matrix of probabilities and computes the ranking
def compute_ranking(y):
    nr_class = y.shape[1]

    y_ranking = np.zeros(list(y.shape))
    for i in range(y.shape[0]):
        ranks = [(y[i, c], c) for c in range(nr_class)]
        ranks.sort()
        ranks.reverse()
        for c in range(nr_class):
            y_ranking[i, c] = ranks[c][1]

    return y_ranking


# Computes sdc, as detailed in the paper mentioned at the beginning of the report
def compute_sdc(mfcc, N = 7, d = 1, p = 3, k = 7):
    n_samples = mfcc.shape[0]
    sdc = np.zeros([n_samples, N * k])

    for t in range(n_samples):
        for coeff in xrange(N):
            for block in xrange(k):
                c_plus = 0
                c_minus = 0

                if t + block * p + d < n_samples:
                    c_plus = mfcc[t + block * p + d][coeff]
                if t + block * p - d >= 0 and t + block * p - d < n_samples:
                    c_minus = mfcc[t + block * p - d][coeff]

                sdc[t][coeff * k + block] = c_plus - c_minus
    return sdc


# Takes in a mp3, converts it to a wav using sox, extract 13 mfcc features using sphinx_fe and finally adds 49 sdc features
def convert_and_extract_features(filename):
    tmp_filename = "tmp/__tmp__.wav"
    feat_filename = "tmp/__tmp__.mfc"
    
    # convert mp3 to wav (16khz, 16 bit, 1 channel)
    os.system("sox {} -R -r 16000 -b 16 -c 1 {}".format(filename, tmp_filename))

    # extract features
    os.system("sphinx_fe -i {} -o {} -ofmt text".format(tmp_filename, feat_filename))

    # enhance features with sdc
    data = np.loadtxt(feat_filename)
    features_62 = np.hstack([data, compute_sdc(data)])
    
    return features_62


# Predicts the result for a mp3. Use return_probs = True if you want the individual gmm model predictions
def predict_mp3(filename, return_probs = False):
    feature_62_gz_filename = "tmp/features_62.gz"
    prediction_filename = "tmp/prediction.out"

    features = convert_and_extract_features(filename)

    np.savetxt(feature_62_gz_filename, features, fmt="%10.6f", comments = "", header = "{} {}".format(features.shape[1], len(features)))

    # load logistic regression calibration model
    logreg = joblib.load("emm/logreg_calibration_final")

    # load inverse language map
    id_to_language = pickle.load(open("data/id_to_language"))

    # predict on NUM_LANGUAGES GMM models
    with open(prediction_filename, 'w') as output:
        for i in tqdm(xrange(NUM_LANGUAGES)):
            logging.info ("Predicting for %s, %s", i, id_to_language[i])

            folder = "emm"
            if return_probs:
                folder = "new_emm"

            server = subprocess.call(["gmmclass", "-d", feature_62_gz_filename, "-m", "{}/{}_{}".format(folder, MODEL_NAME, i), "-t", "4"], stdout = output)
        output.write("finished\n")

    # load prediction in X_inter
    X_inter = np.zeros([1, NUM_LANGUAGES])
    lines = open(prediction_filename).read().splitlines()
    for i in xrange(NUM_LANGUAGES):
        X_inter[0, i] = float(lines[i].split()[-1])
    assert lines[-1] == "finished"

    if return_probs:
        return X_inter
    
    # calibrated prediction scores
    probs = logreg.predict_proba(X_inter)

    # compute ranking
    ranking = compute_ranking (probs)

    result = []
    for i in xrange(3):
        result.append("{},{},{}".format(filename.split(r"/")[-1], id_to_language[ranking[0, i]], i + 1))
    
    logging.info("Prediction result: %s", "\n".join(result))
    return result


# Train the modesl. Takes in a train_truth csv of the form (filename, language) and a relative mp3_dir to prefix
def train_models(train_truth_filename, mp3_dir):
    train_truth = pd.read_csv(train_truth_filename)

    # load language map
    language_to_id = pickle.load(open("data/language_to_id"))

    # load inverse language map
    id_to_language = pickle.load(open("data/id_to_language"))

    # training GMM models
    for lang in tqdm(xrange(NUM_LANGUAGES)):
        start_time = time.time()
        logging.info("Gathering data for language %s, %s", lang, id_to_language[lang])
        lang_dataset = []
        for index, el in train_truth.iterrows():
            if language_to_id[el.language] == lang:
                lang_dataset.append(convert_and_extract_features(mp3_dir + r"/" + el.filename))

        if not len(lang_dataset):
            continue

        lang_dataset = np.vstack(lang_dataset)
        language_gz_filename = "tmp/lang_{}.gz".format(lang)

        np.savetxt(language_gz_filename, lang_dataset, fmt="%10.6f", comments = "", header = "{} {}".format(lang_dataset.shape[1], len(lang_dataset)))
        logging.info("Training %s GMM with %s features", NUM_MIXTURES, NUM_FEATURES)

        os.system("gmmtrain -d {} -m new_emm/{}_{} -s 0.001 -n {} -t 4".format(language_gz_filename, MODEL_NAME, lang, NUM_MIXTURES))

        logging.info("Total time for language %s: %s", lang, time.time() - start_time)
    # predicting for training set

    probs = np.zeros([len(train_truth), NUM_LANGUAGES])
    y_pred = np.zeros(len(train_truth))

    for index, el in train_truth.iterrows():
        probs[index] = predict_mp3(mp3_dir + r"/" + el.filename, True)
        y_pred[index] = language_to_id[el.language]

    # save predicted probabilities for each training sample x each language
    np.save(open("new_emm/final_train_probs", "w"), probs)

    # fitting logistic regression calibration model
    logreg = linear_model.LogisticRegression(C=1e5, random_state=0)
    logreg.fit(probs, y_pred)

    joblib.dump(logreg, "new_emm/logreg_calibration")


# Predict the result for the first K samples from the testing set, in alphabetical order
def predict_first_K(K):
    test_filenames = open("data/testing_list.csv").read().splitlines()

    final_result = []

    for f in tqdm(test_filenames[:K]):
        result = predict_mp3("data/testing/mp3/{}".format(f))
        open("{}_res".format(f), "w").write("\n".join(result))
        final_result.extend(result)

        open("final_result.csv", "w").write("\n".join(final_result))


# Starting point of the program
def main():
    if len(sys.argv) == 1:
        print """Usage: python main.py [init|predict filename|train filename|predict_K filename].
                  
                  python main.py init - initializes the required data
                  python main.py predict filename - predicts the language for a given mp3
                  python main.py train mp3_dir - trains the GMM models using the given mp3_dir
                  python main.py predict_K nr - predicts the result for the first nr samples from the testing set
                  """
        return

    if sys.argv[1] == "init":
        compute_general_dict("data/training_list.csv", "data/testing_list.csv")
    elif sys.argv[1] == "predict":
        predict_mp3(sys.argv[2])
    elif sys.argv[1] == "train":
        train_models("data/training_list.csv", sys.argv[2])
    elif sys.argv[1] == "predict_K":
        predict_first_K(int(sys.argv[2]))


if __name__ == '__main__':
    main()
