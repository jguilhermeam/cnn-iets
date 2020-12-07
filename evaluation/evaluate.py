import xml.etree.ElementTree as ET
from utils import functions as F
from evaluation.metrics import Metrics

def evaluate_results(results,reference_file,attributes):
    reference = F.read_file(reference_file)
    results_stats = {}
    reference_stats = {}
    right_answers = {}
    attr_evaluation = {}
    for attr in attributes:
        attr_evaluation[attr] = Metrics()
        reference_stats[attr] = 0
        results_stats[attr] = 0
        right_answers[attr] = 0

    for result_record, ref in zip(results, reference):
        reference_record = ET.fromstring('<record>'+ref+'</record>')

        for reference_block in reference_record:
            reference_stats[reference_block.tag] += len(reference_block.text.split())

        for result_block in result_record:
            results_stats[result_block.attr] += len(result_block.value.split())

        for result_block in result_record:
            for reference_block in reference_record:
                if result_block.value in F.normalize_str(reference_block.text) and result_block.attr == reference_block.tag:
                    right_answers[result_block.attr] += len(result_block.value.split())
                    break

    for attr in attributes:
        attr_evaluation[attr].precision = right_answers[attr] / results_stats[attr]
        attr_evaluation[attr].recall = right_answers[attr] / reference_stats[attr]
        attr_evaluation[attr].calculate_f_measure()

    print('---------- Results Evaluation Per Attribute ----------')
    print('{:<15} {:<20} {:<20} {:<18}'.format('Attribute', 'Precision', 'Recall', 'F-Measure'))

    for k, v in attr_evaluation.items():
        print('{:<15} {:<20} {:<20} {:<18}'.format(k, v.precision, v.recall, v.f_measure))
