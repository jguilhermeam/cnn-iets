from utils import functions as F
from models.knowledge_base import KnowledgeBase as KB
from blocking import blocking
from cnn import CNN
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
    #records = blocking.extract_blocks(input_file,k_base)

    df = F.get_dataset(kb_file)

    cnn = CNN(k_base,df,4)
    print(cnn.predict("5th avenue"))
