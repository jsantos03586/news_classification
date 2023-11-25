import matplotlib.pyplot as plt
from datetime import datetime
import csv

from functions import constants


def generate_plot_image(docs_number, preq, preq_a, preq_w, drifts, dataset_name,
                        dataset_type, file_name, classifier_type, feature_type, enrichment_type):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(docs_number), preq, label="Prequential")
    ax.plot(range(docs_number), preq_a, label="Prequential Alpha")
    ax.plot(range(docs_number), preq_w, label="Prequential Window")
    for d in drifts:
        plt.axvline(x=d['index'], color='r')
    ax.legend()
    plt.savefig(constants.PLOTS_FOLDER + file_name + '_' + dataset_name + '_' + dataset_type + '_' + classifier_type
                + '_' + feature_type + '_' + enrichment_type + '_' + datetime.now().strftime("%d%m%Y_%I%M%p") + '.png')


def generate_summary_file(docs_total, number_categories, final_accuracy, execution_time, number_drifts, model_summary,
                          dataset_name, dataset_type, file_name, classifier_type, feature_type, enrichment_type):
    with open(constants.SUMMARY_FOLDER+'summary_' + file_name +'_' + dataset_name + '_' + dataset_type + '_'
              + classifier_type + '_' + feature_type + '_' + enrichment_type +  '_' + datetime.now().strftime("%d%m%Y_%I%M%p") + '.csv', 'w',
              newline='') as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(['nº documents', 'categories', 'mean_accuracy', 'time', 'drifts', 'summary'])
        writer.writerow([docs_total, number_categories, final_accuracy, execution_time, number_drifts,
                        model_summary])
    file.close()


def generate_aux_plot_file(preq, preq_a, preq_w, dataset_name, dataset_type, file_name, classifier_type, feature_type, enrichment_type):
    with open(constants.AUX_PLOT_FOLDER +  'plot_aux_' + file_name + '_' + dataset_name + '_' + dataset_type + '_'
              + classifier_type +  '_' + feature_type + '_' + enrichment_type + '_' + datetime.now().strftime("%d%m%Y_%I%M%p") + '.csv', 'w',
              newline='') as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(['preq', 'preq_a', 'preq_w'])
        writer.writerow([preq, preq_a, preq_w])
    file.close()


def generate_tree_file(model, dataset_name, dataset_type, file_name, classifier_type, feature_type, enrichment_type):
    with open(constants.TREES_FOLDER + 'tree_' + file_name + '_' + dataset_name + '_' + dataset_type + '_'
              + classifier_type + '_' + feature_type + '_' + enrichment_type +  '_' + datetime.now().strftime("%d%m%Y_%I%M%p") + '.dot', 'w') as f:
        f.write(str(model.draw()))
