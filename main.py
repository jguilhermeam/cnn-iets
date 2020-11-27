from utils import functions as F
from utils.knowledge_base import KnowledgeBase as KB
from blocking import blocking,block
from cnn_labelling.cnn import CNN
from cnn_labelling import greedy_labelling
from kb_labelling import anchor_labelling
import sys

if __name__ == "__main__":
    try:
        kb_file = sys.argv[1]
        input_file = sys.argv[2]
        #reference_file = sys.argv[3]
    except IndexError as e:
        print('Missing arguments. Parameters must be: knowledge_base input_file')
        sys.exit(1)

    #retrieve knowledge base
    k_base = KB(kb_file)

    #extract blocks from input file
    records = blocking.extract_blocks(input_file,k_base)
    #records = F.manual_block("CHAN DARETTE 13490 Maxella Ave. Marina Del Rey, (310) 301-1004")

    #kb based labelling - detecting anchor blocks
    anchor_labelling.kb_based_labelling(k_base,records,0.9)

    #cnn-based greedy labelling
    cnn = CNN(k_base)
    greedy_labelling.cnn_greedy_labelling(k_base,records,cnn,0.6)

    #debugging
    F.print_blocks(records)
