from kb import knowledge_base as KB
from anchors import blocks
import sys

if __name__ == "__main__":
    try:
        kb_file = sys.argv[1]
        #input_file = sys.argv[2]
        #reference_file = sys.argv[3]
    except IndexError as e:
        print('Missing arguments. Paramaters must be: knowledge_base input reference')
        exit()
    
    #retrieve knowledge base
    kb = KB.get_kb(kb_file)
    
    #retrieve anchor blocks
    sentence = "using machine learning to retrieve medical records john nash science pub 2019 40-50"
    anchors = blocks.get_anchors(sentence,kb,0.9)
    print(anchors)
