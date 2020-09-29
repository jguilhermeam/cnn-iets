from functions import functions as F
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
    kb = F.get_kb(kb_file)
    
    #retrieve anchor blocks
    sentence = "using machine learning to retrieve medical records john nash science pub 2019 40-50"
    p = F.sentence_to_vector(kb,sentence)
    print(p)
