import os
import tempfile
from music21 import converter, midi, stream, key, interval, pitch, note, chord, duration, instrument
import glob
import numpy as np
from sklearn.preprocessing import normalize
import random

def file2extracted(file, melody_idx, harmony_idx):
    s = converter.parse(file)

    # Transpose to C. Assume already in C if fails
    try:
        k = s.flat.getElementsByClass('Key')[0]
        i = interval.Interval(k.tonic, pitch.Pitch('C'))
        s = s.transpose(i)
    except:
        print(f'Error transposing: {file}')

    melody = s.parts[melody_idx]

    print(harmony_idx)
    harmony = s.parts[harmony_idx]

    return melody, harmony

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


midis = []
for file in glob.glob('pop/toto-africa.mid'):
    try:
        melody, harmony = file2extracted(file, 3, 0)
        midis.append((melody, harmony))
    except:
        print(f'Error Parsing: {file}')

IDX2NOTE = list(set(note2str(n) for melody, _ in midis for n in melody.flat.getElementsByClass('Note')))
NOTE2IDX = {n: i for i, n in enumerate(IDX2NOTE)}
IDX2CHORD = list(set(chord2str(c) for _, harmony in midis for c in harmony.flat.getElementsByClass('Chord')))
CHORD2IDX = {c: i for i, c in enumerate(IDX2CHORD)}

N = len(IDX2NOTE)
C = len(IDX2CHORD)
CNNtransition = np.zeros((C, N, N))
CCtransition = np.zeros((C, C))

ni = None
prev_ni = None
ci = None
prev_ci = None
for melody, harmony in midis:
    notes = [(n.offset, 'note', note2str(n)) for n in melody.flat.getElementsByClass('Note')]
    chords = [(c.offset, 'chord', chord2str(c)) for c in harmony.flat.getElementsByClass('Chord')]
    combined = sorted(notes + chords)
    for el in combined:
        if el[1] == 'note':
            prev_ni = ni
            ni = NOTE2IDX[el[2]]
            if ci is None or prev_ni is None:
                continue
            CNNtransition[ci, prev_ni, ni] += 1
        elif el[1] == 'chord':
            prev_ci = ci
            ci = CHORD2IDX[el[2]]
            if prev_ci is None:
                continue
            CCtransition[prev_ci, ci] += 1


# Turn into probabilities (TODO: Fix epsilon normalization)
CCtransition = normalize(CCtransition + 0.001, axis=1, norm='l1')
CNNtransition = np.array([normalize(x, axis=1, norm='l1') for x in CNNtransition + 0.001])


# Generate music!
melody = stream.Stream([instrument.Sampler()])
melody2 = stream.Stream([instrument.Oboe()])
harmony = stream.Stream([instrument.BrassInstrument()])
# Start from C
ni = NOTE2IDX['D']
n2i = NOTE2IDX['D']
ci = CHORD2IDX['A major']
for offset in range(100):
    if offset % 4 == 0: # Update chord
        probs = CCtransition[ci]
        ci = np.random.choice(range(len(probs)), p=probs)
        c = str2chord(IDX2CHORD[ci]).transpose(-12)
        c.duration = duration.Duration(4)
        harmony.insert(offset, c)
    # Update note
    probs = CNNtransition[ci, ni]
    ni = np.random.choice(range(len(probs)), p=probs)
    n = str2note(IDX2NOTE[ni])
    n.duration = duration.Duration(random.choice([0.5, 1, 1.5]))
    melody.insert(offset, n)
    # Update note 2
    probs = CNNtransition[ci, n2i]
    n2i = np.random.choice(range(len(probs)), p=probs)
    n = str2note(IDX2NOTE[n2i])
    n.duration = duration.Duration(random.choice([0.5, 1, 1.5]))
    melody2.insert(offset, n)
s = stream.Stream([melody, harmony])
