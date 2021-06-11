import os
import tempfile
from music21 import converter, midi, stream, key, interval, pitch, note
import glob
import numpy as np
from sklearn.preprocessing import normalize

def file2extracted(file):
    s = converter.parse(file)

    # Transpose to C. Assume already in C if fails
    try:
        k = s.flat.getElementsByClass('Key')[0]
        i = interval.Interval(k.tonic, pitch.Pitch('C'))
        s = s.transpose(i)
    except:
        print(f'Error transposing: {file}')

    melody = s.parts[0]
    return melody

notes = []
for file in glob.glob('final_fantasy/*.mid'):
    try:
        melody = file2extracted(file)
        notes += [n.name for n in melody.getElementsByClass('Note')] # Assumes ordered already
    except:
        print(f'Error Parsing: {file}')

IDX2NOTE = list(set(notes))
NOTE2IDX = {n: i for i, n in enumerate(IDX2NOTE)}

N = len(IDX2NOTE)
transition = np.zeros((N, N))

for n, next_n in zip(notes, notes[1:]):
    transition[NOTE2IDX[n], NOTE2IDX[next_n]] += 1

transition = normalize(transition, axis=1, norm='l1') # Make rows sum to 1




# Make music from transition matrix

s = stream.Stream()
i = 0 # Start from first note
for _ in range(100):
    s.append(note.Note(IDX2NOTE[i]))
    probs = transition[i]
    i = np.random.choice(range(len(probs)), p=probs)
