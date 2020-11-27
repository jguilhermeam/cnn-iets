
def kb_based_labelling(k_base,records,threshold):
    kb = k_base.k_base
    attributes = k_base.get_attributes()
    for blocks in records:
        anchors = {}
        for attr in attributes:
            anchors[attr] = []
        for block in blocks:
            denominator = 1
            for attr in attributes:
                for word in block.value.split():
                    if word in kb[attr]:
                        freq = kb[attr][word]
                        for x in range(1,freq+1):
                            denominator += 1/x
            for attr in attributes:
                numerator = 0
                for word in block.value.split():
                    if word in kb[attr]:
                        freq = kb[attr][word]
                        for x in range(1,freq+1):
                            numerator += 1/x
                if numerator > 0: #this is to avoid false 1.0 probabilities
                    numerator += 1
                prob = (numerator/denominator)
                if prob > threshold:
                    if len(anchors[attr]) > 0:
                        if prob > anchors[attr][0].anchor_prob:
                            anchors[attr][0].clear()
                            block.set_anchor(attr,prob,1)
                            anchors[attr] = [block]
                    else:
                        block.set_anchor(attr,prob,1)
                        anchors[attr].append(block)
        #TODO fix anchors
