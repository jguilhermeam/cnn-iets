from utils import functions as F
from models.knowledge_base import KnowledgeBase as KB
from blocking import blocking
from cnn import CNN
import sys,time

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

    df,code_labels = F.get_dataset(kb_file,num_classes)

    cnn = CNN(k_base,df,num_classes,code_labels)

    print("Making predictions...")
    for r in records:
        F.label_anchor_blocks(k_base,r,0.9)
        probs = []
        missing_anchors = F.get_missing_anchors(r,k_base.k_base)
        print("missing_anchors = "+str(missing_anchors))
        for i,block in enumerate(r):
            if block.is_anchor() == False:
                cnn_output = cnn.predict(block)
                probs.extend(F.adjust_cnn_probs(cnn_output,r,i,missing_anchors))
        print("Greedy labelling...")
        F.greedy_labelling(r,probs,0.6)
        for block in r:
            print(block.value+" - label="+block.label)
        print("\n===================\n")
        time.sleep(10)
