"""Separate the data collection from its processing"""

import torch

TRAINING_DATA = """\
.   -10  -8  -6  -4  -2  0  2  4  6  8 10
100   1  -1  -1  -1  -1 -1 -1 -1 -1 -1 -1
90    1   1  -1  -1  -1 -1 -1 -1 -1 -1 -1
80    1   1  -1  -1  -1 -1 -1 -1 -1 -1 -1
70    1   1   1  -1  -1 -1 -1 -1 -1 -1 -1
60    1   1   1  -1  -1 -1 -1 -1 -1 -1 -1
50    1   1   1  -1  -1 -1 -1 -1 -1 -1 -1
40    1   1   1  -1  -1 -1 -1 -1 -1 -1 -1
30    1   1   1   1  -1 -1 -1 -1 -1 -1 -1
20    1   1   1   1  -1 -1 -1 -1 -1 -1 -1
10    1   1   1   1   1 -1 -1 -1 -1 -1 -1
0     1   1   1   1   1 -1 -1 -1 -1 -1 -1"""

def parse_training_data(data = TRAINING_DATA) :
    rows = data.split('\n')
    colheads = [torch.tensor(float(cell), dtype=torch.float64, requires_grad=True) for cell in rows[0].split()[1:]]
    rowheads = [torch.tensor(float(row.split(None, 1)[0]), dtype=torch.float64, requires_grad=True) for row in rows[1:]]
    inputs = []
    outputs = []
    for rowhead, row in zip(rowheads, rows[1:]):
        for colhead, cell in zip(colheads, row.split()[1:]) :
            inputs.append([rowhead, colhead])
            outputs.append([torch.tensor(float(cell), dtype=torch.float64, requires_grad=True)])
    return inputs, outputs

if __name__ == '__main__':
    inputs, outputs = parse_training_data()
    for a,b in zip(inputs, outputs) :
        print(' '.join(f'{xi:0.1f}' for xi in a), f'{b[0].item():0.1f}')
