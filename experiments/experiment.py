import sys
import time
import logging
import string
import pandas as pd

from river import drift
from river import metrics
from river import feature_extraction as fx
from river import compose
from river import tree
from river import stream

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import ast

from functions import manage_dataset, constants, logging_functions, generate_results

def main():
    logging_functions.create_log_file("experiment_" + str(sys.argv[4]) + '_'+ str(sys.argv[5]) + '_' + str(sys.argv[6]) + '_' + str(sys.argv[7]))
    logging.info('Starting original experiment with: %s', str(sys.argv))
    start_time = time.time()
    page_hinkley = drift.binary.DDM()
    ps = PorterStemmer()
    preds = []
    soma = 0
    preq = []
    drifts = []

    alpha = 0.99
    soma_a = 0
    nr_a = 0
    preq_a = []

    wind = 500
    soma_w = 0
    preq_w = []

    metric = metrics.Accuracy()
    accuracies = []

    dataset = manage_dataset.load_dataset(constants.DATASETS_FOLDER + str(sys.argv[1]))
    dataset['title_stemmed'] = pd.Series(dtype='string')
    dataset['text'] = pd.Series(dtype='string')

    if str(sys.argv[2]) == 'BoW':
        en_stops = set(stopwords.words('english'))
        feature_extraction = fx.BagOfWords(lowercase=True, strip_accents=False, on='text', stop_words=en_stops)
    else:
        feature_extraction = fx.TFIDF(lowercase=True, strip_accents=False, on='text')

    if str(sys.argv[3]) == 'adaptive':
        classifier = tree.HoeffdingAdaptiveTreeClassifier(grace_period=constants.GRACE_PERIOD, delta=constants.DELTA,
                                                          split_criterion=constants.SPLIT_CRITERION, max_depth=1000,
                                                          bootstrap_sampling=False, drift_detector=drift.binary.DDM(),
                                                          nominal_attributes=['category'], seed=0)
    else:
        classifier = tree.HoeffdingTreeClassifier(grace_period=constants.GRACE_PERIOD, delta=constants.DELTA,
                                                  split_criterion=constants.SPLIT_CRITERION, max_depth=1000,
                                                  nominal_attributes=['category'])

    pipeline_original = compose.Pipeline(
        ('feature_extraction', feature_extraction),
        ('classifier', classifier))

    target = dataset['category']
    docs = dataset.drop(['category'], axis=1)

    index = 0

    # Perform the online classification loop
    for xi, yi in stream.iter_pandas(docs, target):
        # Preprocess the current instance
        if str(sys.argv[6]) == 'title':
            text_no_punct = str(xi['title']).translate(str.maketrans("", "", string.punctuation))
        elif str(sys.argv[6]) == 'abstract':
            text_no_punct = str(xi['abstract']).translate(str.maketrans("", "", string.punctuation))
        else:
            text_no_punct = str(xi['title']).translate(str.maketrans("", "", string.punctuation)) + str(xi['abstract']).translate(str.maketrans("", "", string.punctuation))
        word_tokens = word_tokenize(text_no_punct)
        text = []
        for word in word_tokens:
            text.append(ps.stem(word))
        stemming = ' '.join([sub for sub in text])
        logging.info('Index = %s', index)
        logging.info('Title = %s', xi['title'])
        logging.info('Stemming = %s', stemming)
        xi['title_stemmed'] = stemming

        if str(sys.argv[5]) == 'original':
            xi['text'] = stemming
        elif str(sys.argv[5]) == 'enriched':
            if str(sys.argv[6]) == 'title':
                xi['text'] = xi['title_entities']
            elif str(sys.argv[6]) == 'abstract':
                xi['text'] = xi['abstract_entities']
            else:
                xi['text'] = xi['title_entities'] + xi['abstract_entities']   
        else:
            if str(sys.argv[6]) == 'title':
                xi["title_entities"] = ast.literal_eval(xi["title_entities"])
                entities = ' '.join([ sub for sub in xi['title_entities'] ])
                xi['text'] = stemming + entities
            elif str(sys.argv[6]) == 'abstract':
                xi['text'] = stemming + xi['abstract_entities']
            else:
                xi['text'] = stemming + xi['title_entities'] + xi['abstract_entities']   

        pipeline_original['feature_extraction'].learn_one(xi)
        transformed_doc = pipeline_original['feature_extraction'].transform_one(xi)
        logging.info('Feature extraction result = %s', transformed_doc)

        # Make predictions and update the evaluation metric using the classifier
        y_pred = pipeline_original['classifier'].predict_one(transformed_doc)
        metric.update(yi, y_pred)
        accuracies.append(metric.get().real)
        logging.info('Accuracy = %s', metric.get().real)

        if y_pred == yi:
            val = 0
        else:
            val = 1
        preds.append(val)
        soma += val
        preq.append(soma / (index + 1))

        soma_a = val + alpha * soma_a
        nr_a = 1 + alpha * nr_a
        preq_a.append(soma_a / nr_a)

        soma_w += val
        if index >= wind:
            soma_w = soma_w - preds[index - wind]
            preq_w.append(soma_w / 500)
        else:
            preq_w.append(soma_w / (index + 1))

        if str(sys.argv[3]) == 'adaptive':
            detector = pipeline_original['classifier'].drift_detector
        else:
            detector = page_hinkley

        _ = detector.update(val)
        if detector.drift_detected:
            logging.info("Change detected at index %s, input value: %s, predict value %s", index, yi, y_pred)
            drifts.append({"index": index, "input": yi, "predict": y_pred})

        # Update the classifier with the preprocessed features and the true label
        pipeline_original['classifier'].learn_one(transformed_doc, yi)
        index += 1

        try: 
            logging.info('Its possible to print the tree')
        except:
            logging.info('Not possible to print the tree')

    dataset_name = str(sys.argv[4])
    dataset_type = str(sys.argv[5])

    generate_results.generate_plot_image(docs_number=index, preq=preq, preq_a=preq_a, preq_w=preq_w,
                                         drifts=drifts, dataset_name=dataset_name, dataset_type=dataset_type,
                                         file_name='experiment', classifier_type=str(sys.argv[3]), feature_type=str(sys.argv[6]), enrichment_type=str(sys.argv[7]))

    generate_results.generate_summary_file(docs_total=index, number_categories=dataset['category'].nunique(),
                                           final_accuracy=metric.get().real, execution_time=(time.time() - start_time),
                                           number_drifts=len(drifts), model_summary=pipeline_original['classifier'].summary,
                                           dataset_name=dataset_name, dataset_type=dataset_type, file_name='experiment',
                                           classifier_type=str(sys.argv[3]), feature_type=str(sys.argv[6]), enrichment_type=str(sys.argv[7]))

    generate_results.generate_aux_plot_file(preq=preq, preq_a=preq_a, preq_w=preq_w, dataset_name=dataset_name,
                                            dataset_type=dataset_type, file_name='experiment', classifier_type=str(sys.argv[3]),
                                            feature_type=str(sys.argv[6]), enrichment_type=str(sys.argv[7]))

    generate_results.generate_tree_file(model=pipeline_original['classifier'], dataset_name=dataset_name,
                                        dataset_type=dataset_type, file_name='experiment', classifier_type=str(sys.argv[3]),
                                        feature_type=str(sys.argv[6]), enrichment_type=str(sys.argv[7]))

    logging.info("--- %s seconds ---" % (time.time() - start_time))
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    main()



