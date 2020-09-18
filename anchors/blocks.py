def get_anchors(sentence, kb, threshold):
    splited = sentence.split()
    probs = [0 for x in range(len(splited))]
    p_attrs = [None for x in range(len(splited))]
    
    #get most probable attr for each word
    for i,word in enumerate(splited):
        denominator = 1
        for attr in kb:
            if word in kb[attr]:
                freq = kb[attr][word]
                for x in range(1,freq+1):
                    denominator += 1/x
        for attr in kb:
            numerator = 1
            if word in kb[attr]:
                freq = kb[attr][word]
                if freq > 0:
                    for x in range(1,freq+1):
                        numerator += 1/x
                    prob = numerator/denominator
                    if prob > probs[i] and prob > threshold:
                        probs[i] = prob
                        p_attrs[i] = attr
    return fill_spaces(p_attrs)



def fill_spaces(p_attrs):
    #join blocks from between words of the same attribute
    current = p_attrs[0]
    c_i = 0
    for i in range(1,len(p_attrs)):
        if p_attrs[i] == current:
            for j in range(i,c_i,-1):
                p_attrs[j] = current
        else:
            if p_attrs[i] != None:
                current = p_attrs[i]
                c_i = i

    return p_attrs
