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
    num_classes = len(k_base.k_base)
    records = blocking.extract_blocks(input_file,k_base)

    F.label_anchor_blocks(k_base,records,0.9)

    df,code_labels = F.get_dataset(kb_file,num_classes)

    cnn = CNN(k_base,df,num_classes,code_labels)

    print("Making predictions...")
    for r in records:
        probs = []
        for block in r:
            probs.extend(cnn.predict(block))
        print("Greedy labelling...")
        print("Probs:")
        for x in probs:
            print("segment "+x[1].value+" in label="+x[2]+" - "+str(x[0]))
        F.greedy_labelling(r,probs,0.3)
        for block in r:
            print(block.value+" - label="+block.label)
        exit()
