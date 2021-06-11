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
    return n.name

def str2note(s):
    return note.Note(s)

def chord2str(c):
    return ' '.join([c.root().name, c.quality])

def str2chord(s):
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

def debounce_chords(chords):
    new_chords = []
    i = 0
    while i < len(chords):
        c = chords[i]
        j = i + 1
        # Find first different chord
        while j < len(chords) and chord2str(c) == chord2str(chords[j]):
            j += 1
        c.offset = i
        c.duration = duration.Duration(j-i)
        new_chords.append(c)
        i = j
    return new_chords

def debounce_notes(notes):
    new_notes = []
    i = 0
    while i < len(notes):
        n = notes[i]
        j = i + 1
        # Find first different note
        while j < len(notes) and note2str(n) == note2str(notes[j]):
            j += 1
        n.offset = i
        n.duration.quarterLength = j-i
        new_notes.append(n)
        i = j
    return new_notes



# Choose song
song = 'final_fantasy/ff1prologue.mid'
# song = 'pop/alan_jackson-gone_country.mid'
s = converter.parse(song)
score_tempo = s.flat.getElementsByClass('MetronomeMark')[0]

# Raw notes
notes = s.parts[0].flat.getElementsByClass('Note')
chords = s.parts[3].flat.getElementsByClass('Chord')
# notes = debounce_notes(s.flat.getElementsByClass('Note'))
# chords = debounce_chords(s.flat.getElementsByClass('Chord'))


combined = sorted(list(notes) + list(chords), key=lambda x: (x.offset, isinstance(x, note.Note)))

IDX2NOTE = list(set(note2str(n) for n in notes))
NOTE2IDX = {n: i for i, n in enumerate(IDX2NOTE)}
N = len(IDX2NOTE)
IDX2CHORD = list(set(chord2str(c) for c in chords))
CHORD2IDX = {c: i for i, c in enumerate(IDX2CHORD)}
C = len(IDX2CHORD)


# Build encoding
ci = 0
ni = 0
encoded = torch.zeros((len(combined), C + N))
for idx, e in enumerate(combined):
    if isinstance(e, chord.Chord):
        ci = CHORD2IDX[chord2str(e)]
    elif isinstance(e, note.Note):
        ni = NOTE2IDX[note2str(e)]
    encoded[idx][ci] = 1
    encoded[idx][C + ni] = 1


# Build Training Set
max_seq_len = 7
train_data = []
for seq_len in range(1, max_seq_len):
    for i in range(len(encoded) - seq_len):
        seq = encoded[i:i+seq_len]
        label = encoded[i+seq_len]
        train_data.append((seq, label))



# RNN
init_hidden = (torch.zeros(1, 1, C+N), torch.zeros(1, 1, C+N)) # Why does it need so many extra dimensions??
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(C + N, C + N)

    def forward(self, input, hidden):
        input = input.view(1,1,-1)
        output, hidden = self.lstm(input, hidden)
        return output.view(-1), hidden

net = RNN()



# Train RNN
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.15, momentum=0.3, nesterov=True, weight_decay=2e-4)
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
ci = 0
ni = 0
hidden = init_hidden
chords = []
notes = []
with torch.no_grad():
    for i in range(100):
        # if i % 10 == 0:
        #     hidden = init_hidden
        token = torch.zeros(C+N); token[ci] = 1; token[C+ni] = 1
        output, hidden = net(token, hidden)
        chord_probs = F.softmax(output[:C]).numpy()
        note_probs = F.softmax(output[C:]).numpy()
        ci = np.random.choice(range(len(chord_probs)), p=chord_probs)
        ni = np.random.choice(range(len(note_probs)), p=note_probs)
        chords.append(str2chord(IDX2CHORD[ci]))
        notes.append(str2note(IDX2NOTE[ni]))



# Structure the music
harmony = stream.Stream(debounce_chords(chords))
melody = stream.Stream(debounce_notes(notes))
title = f"LSTM Inspired by {song.split('/')[-1]}"
s = stream.Stream([metadata.Metadata(title=title), score_tempo, melody, harmony])
s.write('midi', title.replace(' ', '_'))
s.write('mxl', title.replace(' ', '_') + '.mxl')
