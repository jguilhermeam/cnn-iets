import sys
import unicodedata
import re
import time
import xml.etree.ElementTree as ET
from blocking.block import Block

def read_file(filename):
    try:
        with open(filename, 'r') as f:
            return f.read().splitlines()
    except IOError as error:
        print("[ERROR] file not found - "+filename)
        sys.exit(1)

def normalize_str(input_str):
    '''Transform string to lowercase, remove special chars,
    accents and trailing spaces'''
    byte_string = unicodedata.normalize('NFD', input_str.lower()).encode('ASCII', 'ignore')
    normalized = re.sub(r'[^A-Za-z0-9\s]+', '',byte_string.decode('utf-8')).strip()
    #if re.match(r'^\d+\s\d+$', normalized):
    #    normalized = re.sub(r' ', '', normalized)
    normalized = re.sub('\d','dg',normalized) #change digits do dg
    return normalized

#functions only for debugging purposes
def manual_block(text):
    blocks_list = []
    for word in text.split():
        blocks_list.append(Block(normalize_str(word), word))
    return [blocks_list]

def print_blocks(records):
    for blocks in records:
        for block in blocks:
            print(block.raw_value+" - attribute="+block.attr)
        print("\n===================\n")
        time.sleep(12)
