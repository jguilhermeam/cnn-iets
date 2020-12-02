def cnn_greedy_labelling(k_base,records,cnn,threshold):
    for blocks in records:
        probabilities = []
        for i in range(0,len(blocks)):
            if blocks[i].is_anchor == False:
                cnn_output = cnn.predict(blocks[i])
                probabilities.extend(normalize_cnn_probabilities(k_base,cnn_output,blocks,i))
        greedy_labelling(blocks,probabilities,threshold)

def get_missing_anchors(record,k_base):
    missing = k_base.get_attributes()
    for block in record:
        if block.is_anchor == True:
            missing.remove(block.attr)
    return missing

def normalize_cnn_probabilities(k_base,probabilities,blocks,pos):
    attributes = k_base.get_attributes()
    missing_anchors = get_missing_anchors(blocks,k_base)
    possible_attributes = []
    for j in reversed(range(0,pos)):
        if blocks[j].is_anchor == True:
            possible_attributes.append(blocks[j].attr)
            break
    for j in range(pos+1,len(blocks)):
        if blocks[j].is_anchor == True:
            possible_attributes.append(blocks[j].attr)
            break
    possible_attributes.extend(missing_anchors)
    normalized_probabilities = []
    denominator = 0
    for attr in possible_attributes:
        denominator += probabilities[attr]
    for attr in possible_attributes:
        p = probabilities[attr]/denominator
        blocks[pos].cnn_scores[attr] = p
        normalized_probabilities.append((p,blocks[pos],attr))
    for attr in attributes:
        if attr not in blocks[pos].cnn_scores:
            blocks[pos].cnn_scores[attr] = 0
    return normalized_probabilities


def get_max_prob(partition,probabilities):
    max = -1
    choosen = None
    for p in probabilities:
        if p[0] >= max and p[1] in partition:
            max = p[0]
            choosen = p
    return choosen

def greedy_labelling(blocks,probabilities,threshold):
    partitions = [[blocks[0]]]
    j = 0
    for i in range(1,len(blocks)):
        if blocks[i].is_anchor == True and blocks[i-1].is_anchor == True:
            partitions.append([blocks[i]])
            j += 1
        else:
            partitions[j].append(blocks[i])
    for p in partitions:
        while True:
            choosen = get_max_prob(p,probabilities)
            if choosen == None or choosen[0] <= threshold:
                break
            probabilities.remove(choosen)
            sc = choosen[1]
            Ac = choosen[2]
            if sc.attr != 'none':
                continue
            sc.attr = Ac
            index = blocks.index(sc)
            for i,sc_b in enumerate(blocks):
                if i < index and sc_b.attr == Ac:
                    in_between = False
                    for j in range(i+1,index):
                        if blocks[j].attr != Ac and blocks[j].attr != 'none':
                            in_between = True
                    if in_between == True:
                        sc.attr = 'none'
                    else:
                        for j in range(i+1,index):
                            blocks[j].attr = Ac
                elif i > index and sc_b.attr == Ac:
                    in_between = False
                    for j in range(index+1,i):
                        if blocks[j].attr != Ac and blocks[j].attr != 'none':
                            in_between = True
                    if in_between == True:
                        sc.attr = 'none'
                    else:
                        for j in range(index+1,i):
                            blocks[j].attr = Ac
            if len(probabilities) == 0:
                break
            exists_unlabelled = False
            for b in p:
                if b.attr == 'none':
                    exists_unlabelled = True
                    break
            if exists_unlabelled == False:
                break
