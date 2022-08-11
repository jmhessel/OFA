import os
import numpy as np

fs = [x for x in os.listdir('.') if '.log' in x]

records = []
for fn in fs:
    with open(fn) as f:
        best_epoch, best_val_loss = -1, np.inf
        for line in f:
            if "valid on 'valid' subset" in line:
                toks = line.split('|')
                valid_loss = float(toks[2].split()[1])
                valid_epoch = int(toks[0].split()[-1])

                if valid_loss < best_val_loss:
                    best_val_loss = valid_loss
                    best_epoch = valid_epoch
        records.append((fn, best_val_loss, best_epoch))

#'50_5e-6.log'
records = sorted(records, key=lambda x: float(x[0].split('_')[1].split('.')[0]))

for r in records:
    print(r)
