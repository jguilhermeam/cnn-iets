import xml.etree.ElementTree as ET
from utils import functions as F
from evaluation.metrics import Metrics

def evaluate_results(results,reference_file,attributes):
    evaluate_results_per_attribute(results,reference_file,attributes)
    #evaluate_results_per_record(results,reference_file,attributes)

def evaluate_results_per_attribute(results,reference_file,attributes):
    reference = F.read_file(reference_file)
    results_stats = {}
    reference_stats = {}
    right_answers = {}
    attr_evaluation = {}
    record_evaluation = []
    for attr in attributes:
        attr_evaluation[attr] = Metrics()

    for result_record, ref in zip(results, reference):
        reference_record = ET.fromstring('<record>'+ref+'</record>')

        for reference_block in reference_record:
            if reference_block.tag not in reference_stats:
                reference_stats[reference_block.tag] = len(reference_block.text.split())
            else:
                reference_stats[reference_block.tag] += len(reference_block.text.split())

        for result_block in result_record:
            if result_block.attr != 'none':
                if result_block.attr not in results_stats:
                    results_stats[result_block.attr] = len(result_block.value.split())
                else:
                    results_stats[result_block.attr] += len(result_block.value.split())

        for result_block in result_record:
            for reference_block in reference_record:
                if result_block.value in F.normalize_str(reference_block.text) and result_block.attr == reference_block.tag:
                    if result_block.attr not in right_answers:
                        right_answers[result_block.attr] = len(result_block.value.split())
                    else:
                        right_answers[result_block.attr] += len(result_block.value.split())
                    break


    for attr in attributes:
        if attr in results_stats and attr in reference_stats and attr in right_answers:
            attr_evaluation[attr].precision = right_answers[attr] / results_stats[attr]
            attr_evaluation[attr].recall = right_answers[attr] / reference_stats[attr]
            attr_evaluation[attr].calculate_f_measure()

    print()
    print('---------- Results Evaluation Per Attribute ----------')
    print('{:<15} {:<20} {:<20} {:<18}'.format('Attribute', 'Precision', 'Recall', 'F-Measure'))

    total_metrics = Metrics()
    non_zero_attrs = 0
    for k, v in attr_evaluation.items():
        if v.f_measure > 0:
            print('{:<15} {:<20} {:<20} {:<18}'.format(k, v.precision, v.recall, v.f_measure))
            total_metrics.precision += v.precision
            total_metrics.recall += v.recall
            total_metrics.f_measure += v.f_measure
            non_zero_attrs += 1

    total_metrics.precision /= non_zero_attrs
    total_metrics.recall /= non_zero_attrs
    total_metrics.f_measure /= non_zero_attrs
    print()
    print('{:<15} {:<20} {:<20} {:<18}'.format("Total", total_metrics.precision, total_metrics.recall, total_metrics.f_measure))
    print()

def evaluate_results_per_record(results,reference_file,attributes):
    reference = F.read_file(reference_file)
    record_evaluation = []

    for result_record, ref in zip(results, reference):
        results_stats = {}
        reference_stats = {}
        right_answers = {}
        attr_evaluation = {}

        reference_record = ET.fromstring('<record>'+ref+'</record>')

        for reference_block in reference_record:
            if reference_block.tag not in reference_stats:
                reference_stats[reference_block.tag] = len(reference_block.text.split())
            else:
                reference_stats[reference_block.tag] += len(reference_block.text.split())

        for result_block in result_record:
            if result_block.attr != 'none' and result_block.attr not in results_stats:
                results_stats[result_block.attr] = len(result_block.value.split())
            else:
                results_stats[result_block.attr] += len(result_block.value.split())

        for result_block in result_record:
            for reference_block in reference_record:
                if result_block.value in F.normalize_str(reference_block.text) and result_block.attr == reference_block.tag:
                    if result_block.attr not in right_answers:
                        right_answers[result_block.attr] = len(result_block.value.split())
                    else:
                        right_answers[result_block.attr] += len(result_block.value.split())
                    break

        for attr in attributes:
            if attr in results_stats and attr in reference_stats and attr in right_answers:
                attr_evaluation[attr] = Metrics()
                attr_evaluation[attr].precision = right_answers[attr] / results_stats[attr]
                attr_evaluation[attr].recall = right_answers[attr] / reference_stats[attr]
                attr_evaluation[attr].calculate_f_measure()
            elif attr in results_stats and attr not in reference_stats:
                attr_evaluation[attr] = Metrics()

        record = Metrics()
        for attr in attr_evaluation:
            record.precision += attr_evaluation[attr].precision
            record.recall += attr_evaluation[attr].recall
            record.f_measure += attr_evaluation[attr].f_measure
        record.precision /= len(attr_evaluation)
        record.recall /= len(attr_evaluation)
        record.f_measure /= len(attr_evaluation)
        record_evaluation.append(record)

    final_metrics = Metrics()
    for record in record_evaluation:
        final_metrics.precision += record.precision
        final_metrics.recall += record.recall
        final_metrics.f_measure += record.f_measure
    final_metrics.precision /= len(record_evaluation)
    final_metrics.recall /= len(record_evaluation)
    final_metrics.f_measure /= len(record_evaluation)

    print('---------- Results Evaluation Per Record ----------')
    print('{:<20} {:<20} {:<18}'.format('Precision', 'Recall', 'F-Measure'))
    print('{:<20} {:<20} {:<18}'.format(final_metrics.precision, final_metrics.recall, final_metrics.f_measure))
    print()
