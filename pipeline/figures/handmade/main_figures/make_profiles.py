"""
The worked response (hope|298, Claude)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
B = 10
LABELS = json.loads((HERE / 'data/response.json').read_text())['labels']
LI = {l: i for i, l in enumerate(LABELS)}


def tables(df):
    P, T = np.zeros((len(LABELS), B)), np.zeros((len(LABELS), len(LABELS)))
    for _, g in df.groupby(['corpus', 'row']):
        g = g.sort_values('position', kind='stable')
        slots = []
        for x, l in zip(g.position, g.label):
            P[LI[l], min(int(x * B), B - 1)] += 1
            if slots and slots[-1][0] == x:
                slots[-1][1].append(l)
            else:
                slots.append((x, [l]))
        for (_, prev), (_, cur) in zip(slots, slots[1:]):
            T[LI[prev[-1]], LI[cur[0]]] += 1
    return P, T


events = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else HERE / '../../SCRIPT/data/span_events.csv')
worked = (events.corpus == 'hope') & (events.row == 298) & (events.model == 'Claude')
out = {}
for model, g in events[~worked].groupby('model'):
    P, T = tables(g)
    out[model] = {'P': P.tolist(), 'T': T.tolist()}
(HERE / 'data/profiles.json').write_text(json.dumps(out))
print('wrote data/profiles.json for', ', '.join(out))
