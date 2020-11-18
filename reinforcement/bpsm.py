class BPSM:

    def __init__(self, matching_list, k_base):
        self.f_matrix = self.init_f_matrix(
            k_base.get_attributes(), matching_list)
        self.b_matrix = self.init_b_matrix(
            k_base.get_attributes(), matching_list)
        self.p_matrix = self.init_p_matrix(
            k_base.get_attributes(), matching_list)

    def init_f_matrix(self, attribute_list, matching_list):
        attribute_list.append('none')
        attribute_index = {}
        for i, attr in enumerate(attribute_list):
            attribute_index[attr] = {'id': i + 1, 'transitions': 0}

        matrix_size = len(attribute_list)+1
        t_matrix = [[0 for i in range(matrix_size)]
                    for j in range(matrix_size)]

        for i in range(matrix_size-1):
            t_matrix[0][i+1] = attribute_list[i]
            t_matrix[i+1][0] = attribute_list[i]

        # Set the number of transitions from label i to label j
        for block in matching_list:
            for n in range(len(block)-1):
                current_label = block[n].label
                next_label = block[n+1].label
                i = attribute_index[current_label]['id']
                j = attribute_index[next_label]['id']
                attribute_index[current_label]['transitions'] += 1
                t_matrix[i][j] += 1

        # Divides the number of transitions from label i to label j
        # by the total number of transitions starting from label i
        for i in range(matrix_size-1):
            for j in range(matrix_size-1):
                if t_matrix[i+1][j+1] > 0:
                    t_matrix[i+1][j+1] /= attribute_index[t_matrix[i+1]
                                                          [0]]['transitions']

        return t_matrix

    def init_b_matrix(self, attribute_list, matching_list):
        '''Initialize transitions matrix'''
        attribute_list.append('none')
        attribute_index = {}
        for i, attr in enumerate(attribute_list):
            attribute_index[attr] = {'id': i + 1, 'transitions': 0}

        matrix_size = len(attribute_list)+1
        t_matrix = [[0 for i in range(matrix_size)]
                    for j in range(matrix_size)]

        for i in range(matrix_size-1):
            t_matrix[0][i+1] = attribute_list[i]
            t_matrix[i+1][0] = attribute_list[i]

        # Set the number of transitions from label i to label j
        for block in matching_list:
            for n in range(len(block)-1):
                current_label = block[n].label
                next_label = block[n+1].label
                i = attribute_index[current_label]['id']
                j = attribute_index[next_label]['id']
                attribute_index[current_label]['transitions'] += 1
                t_matrix[i][j] += 1

        # Divides the number of transitions from label i to label j
        # by the total number of transitions starting from label i
        for i in range(matrix_size-1):
            for j in range(matrix_size-1):
                if t_matrix[i+1][j+1] > 0:
                    t_matrix[i+1][j+1] /= attribute_index[t_matrix[i+1]
                                                          [0]]['transitions']

        return t_matrix

    def init_p_matrix(self, attribute_list, matching_list):
        '''Initialize positioning matrix'''
        attribute_list.append('none')
        attribute_index = {}
        for i, attr in enumerate(attribute_list):
            attribute_index[attr] = i + 1

        rows = len(attribute_list)+1
        cols = max([len(v) for v in matching_list])+1
        p_matrix = [[0 for i in range(cols)] for j in range(rows)]
        position_index = [{i+1: 0} for i in range(cols-1)]

        for i in range(rows-1):
            p_matrix[i+1][0] = attribute_list[i]
        for i in range(cols-1):
            p_matrix[0][i+1] = (i+1)

        for block in matching_list:
            for n in range(len(block)):
                current_label = block[n].label
                i = attribute_index[current_label]
                j = n+1
                p_matrix[i][j] += 1
                position_index[n][j] += 1

        for i in range(rows-1):
            for j in range(cols-1):
                if p_matrix[i+1][j+1] > 0:
                    p_matrix[i+1][j+1] /= position_index[j][j+1]

        return p_matrix
