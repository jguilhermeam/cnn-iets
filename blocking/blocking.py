from blocking.block import Block
from utils import functions as F


def extract_blocks(input_file, k_base):
    r_input = [r for r in F.read_file(input_file)]
    blocks = []
    for raw_terms in r_input:
        blocks.append(build_blocks(raw_terms, k_base))
    return blocks

def build_blocks(record, k_base):
    segments = record.split(",")
    blocks_list = []
    for b in segments:
        blocks_list.append(Block(F.normalize_str(b), b))
    return blocks_list
#
# def extract_blocks(input_file, k_base):
#     print("Extracting blocks...")
#     r_input = [r for r in F.read_file(input_file)]
#     normalized_input = [F.normalize_str(v) for v in r_input]
#     blocks = []
#     for raw_terms, record in zip(r_input, normalized_input):
#         blocks.append(build_blocks(record.split(), raw_terms.split(), k_base))
#     return blocks
#
#
# def build_blocks(terms, raw_terms, k_base):
#     '''Build a set of blocks for a string'''
#     blocks_list = []
#     blocks_list.append(Block(terms[0], raw_terms[0]))
#     i = 0
#     j = 1
#     while j < len(terms):
#         co_occur = False
#         for entry in k_base.registers:
#             if terms[j-1]+' '+terms[j] in entry:
#                 co_occur = True
#                 break
#         if co_occur == False:
#             blocks_list.append(Block('', ''))
#             i += 1
#         if blocks_list[i].value in '':
#             blocks_list[i].value += terms[j]
#             blocks_list[i].raw_value += raw_terms[j]
#         else:
#             blocks_list[i].value += ' ' + terms[j]
#             blocks_list[i].raw_value += ' ' + raw_terms[j]
#         j += 1
#     return blocks_list
