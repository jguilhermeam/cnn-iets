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
    
    #pandas test
    x = F.get_dataset(kb_file)
    #get embedding matrix
    #a,b = F.kb_to_embedding_matrix(kb,kb_file)
    #print(a[:10])
    #print(b[:10])
