import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import tempfile
from music21 import converter, midi, stream, key, interval, pitch, note, chord, duration, instrument, metadata
import glob
import numpy as np
from sklearn.preprocessing import normalize
import random

def note2str(n):
    return 'NOTE:' + n.name

def chord2str(c):
    return 'CHORD:' + ' '.join([c.root().name, c.quality])

def str2elem(s):
    typ, s = s.split(':', 1)
    if typ == 'NOTE':
        return note.Note(s)
    elif typ == 'CHORD':
        root, quality = s.split(' ')
        root = note.Note(root + '3')
        if quality == 'major':
            c = chord.Chord([root, root.transpose(4), root.transpose(7)])
        elif quality == 'minor':
            c = chord.Chord([root, root.transpose(3), root.transpose(7)])
        else:
            # raise Exception(f'Unimplemented quality {quality}')
            c = chord.Chord([root, root.transpose(4), root.transpose(7)]) # TODO: handle others
        c.volume = 0.4
        return c
    else:
        raise Exception(f'Unexpected type: {typ}')


# Choose song
song = 'final_fantasy/ff1prologue.mid'
s = converter.parse(song)
score_tempo = s.flat.getElementsByClass('MetronomeMark')[0]

# Raw notes
notes = s.parts[0].flat.getElementsByClass('Note')
chords = s.parts[3].flat.getElementsByClass('Chord')

notes = [(n.offset, note2str(n)) for n in notes]
chords = [(c.offset, chord2str(c)) for c in chords]
combined = sorted(notes + chords)

IDX2ELEM = list(set(e[1] for e in combined))
ELEM2IDX = {e: i for i, e in enumerate(IDX2ELEM)}
N = len(IDX2ELEM) # Vocab size

# One-hot encoding
encoded = torch.zeros((len(combined), N))
for idx, e in enumerate(combined):
    encoded[idx][ELEM2IDX[e[1]]] = 1



# Build Training Set
seq_len = 5
train_data = []
for i in range(len(encoded) - seq_len):
    seq = encoded[i:i+seq_len]
    label = encoded[i+seq_len]
    train_data.append((seq, label))



# RNN
HIDDEN_SIZE = 10
init_hidden = torch.zeros(HIDDEN_SIZE)
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.i2h = nn.Linear(N + HIDDEN_SIZE, HIDDEN_SIZE)
        self.i2o = nn.Linear(N + HIDDEN_SIZE, N)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 0).unsqueeze(0)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        return output.squeeze(0), hidden.squeeze(0)

net = RNN()




# Train RNN
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.15, momentum=0.3, nesterov=True, weight_decay=0)
for epoch in range(25):
    print(f"Epoch {epoch}")
    epoch_loss = 0
    for seq, label in train_data:
        optimizer.zero_grad()
        # Feed RNN
        hidden = init_hidden
        for token in seq:
            output, hidden = net(token, hidden)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Avg Epoch Loss: {epoch_loss / len(train_data)}")

# Use RNN to generate list of elements
token = torch.zeros(N); token[0] = 1  # Start with arbitrary one-hot token
hidden = init_hidden
elements = []
with torch.no_grad():
    for _ in range(100):
        output, hidden = net(token, hidden)
        probs = F.softmax(output).numpy()
        idx = np.random.choice(range(len(probs)), p=probs)
        token = torch.zeros(N); token[idx] = 1
        elements.append(str2elem(IDX2ELEM[idx]))


# Structure the music
title = f"RNN Inspired by {song.split('/')[-1]}"
s = stream.Stream([metadata.Metadata(title=title), score_tempo])
for idx, e in enumerate(elements):
    s.insert(idx, e)

s.write('midi', title.replace(' ', '_'))
s.write('mxl', title.replace(' ', '_') + '.mxl')
