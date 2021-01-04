class BPSM:

    def __init__(self, records, attribute_list):
        print("Calculating BPSM matrices...")
        attribute_list.extend(['begin','end'])
        self.f_matrix = self.init_f_matrix(records, attribute_list)
        self.b_matrix = self.init_b_matrix(records, attribute_list)

        attribute_list.remove('end')
        attribute_list.remove('begin')
        self.p_matrix = self.init_p_matrix(records, attribute_list)

    def init_f_matrix(self, records, attribute_list):
        transitions = {}
        matrix = {}
        for i in attribute_list:
            transitions[i] = 0
            matrix[i] = {}
            for j in attribute_list:
                matrix[i][j] = 0

        for blocks in records:
            for aux in range(len(blocks)):
                if aux == 0:
                    i = 'begin'
                else:
                    i = blocks[aux-1].attr
                j = blocks[aux].attr
                if i != 'none' and j != 'none' and i != j:
                    matrix[i][j] += 1
                    transitions[i] += 1

        for i in attribute_list:
            for j in attribute_list:
                if transitions[i] > 0:
                    matrix[i][j] /= transitions[i]

        return matrix

    def init_b_matrix(self, records, attribute_list):
        transitions = {}
        matrix = {}
        for k in attribute_list:
            transitions[k] = 0
            matrix[k] = {}
            for j in attribute_list:
                matrix[k][j] = 0

        for blocks in records:
            for aux in reversed(range(len(blocks))):
                if aux == len(blocks)-1:
                    k = 'end'
                else:
                    k = blocks[aux+1].attr
                j = blocks[aux].attr
                if k != 'none' and j != 'none' and k != j:
                    matrix[k][j] += 1
                    transitions[k] += 1

        for k in attribute_list:
            for j in attribute_list:
                if transitions[k] > 0:
                    matrix[k][j] /= transitions[k]

        return matrix

    def init_p_matrix(self, records, attribute_list):
        max_positions = max([len(v) for v in records])
        total_in_pos = [0 for i in range(max_positions)]
        matrix = {}
        for attr in attribute_list:
            matrix[attr] = [0 for i in range(max_positions)]

        for blocks in records:
            for i in range(0,len(blocks)):
                attr = blocks[i].attr
                if attr != 'none':
                    matrix[attr][i] += 1
                    total_in_pos[i] += 1

        for attr in attribute_list:
            for i in range(max_positions):
                if total_in_pos[i] > 0:
                    matrix[attr][i] /= total_in_pos[i]

        return matrix
