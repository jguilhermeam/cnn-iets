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
            #begin
            i = 'begin'
            j = blocks[0].attr
            if j != 'none':
                matrix[i][j] += 1
                transitions[i] += 1
            #middle
            last = len(blocks)-1
            for aux in range(0,last):
                i = blocks[aux].attr
                j = blocks[aux+1].attr
                if i != 'none' and j != 'none':
                    matrix[i][j] += 1
                    transitions[i] += 1
            #ending
            i = blocks[last].attr
            j = 'end'
            if i != 'none':
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
        for i in attribute_list:
            transitions[i] = 0
            matrix[i] = {}
            for j in attribute_list:
                matrix[i][j] = 0

        for blocks in records:
            #ending
            last = len(blocks)-1
            k = 'end'
            j = blocks[last].attr
            if j != 'none':
                matrix[k][j] += 1
                transitions[k] += 1
            #middle
            for aux in reversed(range(1,last)):
                k = blocks[aux].attr
                j = blocks[aux-1].attr
                if k != 'none' and j != 'none':
                    matrix[k][j] += 1
                    transitions[k] += 1
            #begin
            k = blocks[0].attr
            j = 'begin'
            if k != 'none':
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
                matrix[attr][i] /= total_in_pos[i]

        return matrix
