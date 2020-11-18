from blocking.block import Block
from utils import functions as F


def extract_blocks(input_file, k_base):
    print("Extracting blocks...")
    r_input = [r for r in F.read_file(input_file)]
    normalized_input = [F.normalize_str(v) for v in r_input]
    blocks = []
    for raw_terms, record in zip(r_input, normalized_input):
        blocks.append(build_blocks(record.split(), raw_terms.split(), k_base))
    return blocks


def build_blocks(terms, raw_terms, k_base):
    '''Build a set of blocks for a string'''
    blocks_list = []
    blocks_list.append(Block(terms[0], raw_terms[0]))
    i = 0
    j = 1
    while j < len(terms):
        if not co_occurs(terms[j-1], terms[j], k_base):
            blocks_list.append(Block('', ''))
            i += 1
        if blocks_list[i].value in '':
            blocks_list[i].value += terms[j]
            blocks_list[i].raw_value += raw_terms[j]
        else:
            blocks_list[i].value += ' ' + terms[j]
            blocks_list[i].raw_value += ' ' + raw_terms[j]
        j += 1
    return blocks_list


def co_occurs(current_term, next_term, k_base):
    '''Verify if the current term and next term are known
    to co-occur in some occurrence in the knowledge base'''
    if current_term in k_base.inverted_k_base and next_term in k_base.inverted_k_base:
        co_occurrences = k_base.co_occurrences[current_term]
        if next_term in [x[0] for x in co_occurrences]:
            return True
    return False

def merge_blocks(aux):
    block = aux[0]
    for i in range(1,len(aux)):
        block.value += ' ' + aux[i].value
        block.raw_value += ' ' + aux[i].raw_value
    return block

def join_blocks(record):
    aux = []
    final_record = []
    for block in record:
        if block.label != 'none':
            if len(aux) == 0 or block.label == aux[0].label:
                aux.append(block)
            else:
                final_record.append(merge_blocks(aux))
                aux = [block]
        else:
            if len(aux) > 0:
                final_record.append(merge_blocks(aux))
                aux = []
            final_record.append(block)
    if len(aux) > 0:
        final_record.append(merge_blocks(aux))
    return final_record
