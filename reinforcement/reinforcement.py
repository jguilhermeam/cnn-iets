from reinforcement.bpsm import BPSM

def reinforce(records, attribute_list):
    bpsm = BPSM(records, attribute_list)
    print("applying reinforcement...")

    for blocks in records:
        i = 0
        k = len(blocks)-1

        while True:
            if i >= k:
                break
            prob_fw = calculate_pfw(i,blocks,attribute_list,bpsm)
            prob_bw = calculate_pbw(k,blocks,attribute_list,bpsm)

            if prob_fw[0] >= prob_bw[0]:
                blocks[i].attr = prob_fw[1]
                i += 1
            else:
                blocks[k].attr = prob_bw[1]
                k -= 1



def calculate_pfw(j,blocks,attribute_list,bpsm):
    if j == 0:
        i_attr = 'begin'
    else:
        i_attr = blocks[j-1].attr

    attribute_score = {}
    for attr in attribute_list:
        attribute_score[attr] = 1 - ((1-blocks[j].cnn_scores[attr])*(1-bpsm.f_matrix[i_attr][attr])*(1-bpsm.p_matrix[attr][j]))
    blocks[j].fw_reinforcement_score = attribute_score
    return blocks[j].get_top_fw_reinforcement_score()

def calculate_pbw(j,blocks,attribute_list,bpsm):
    last = len(blocks)-1
    if j == last:
        k_attr = 'end'
    else:
        k_attr = blocks[j+1].attr

    attribute_score = {}
    for attr in attribute_list:
        attribute_score[attr] = 1 - ((1-blocks[j].cnn_scores[attr])
        *(1-bpsm.b_matrix[k_attr][attr])
        *(1-bpsm.p_matrix[attr][j]))
    blocks[j].bw_reinforcement_score = attribute_score
    return blocks[j].get_top_bw_reinforcement_score()
