from kb import knowledge_base as KB
import sys

if __name__ == "__main__":
    try:
        kb_file = sys.argv[1]
        #input_file = sys.argv[2]
        #reference_file = sys.argv[3]
    except IndexError as e:
        print('Missing arguments. Paramaters must be: knowledge_base input reference')
        exit()
    
    kb = KB.get_kb(kb_file)
    for x in kb:
        print(x)
