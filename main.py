from functions import functions as F
from cnn import CNN
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
    
    df = F.get_dataset(kb_file)
    cnn = CNN(kb)
    cnn.preprocess(df)
