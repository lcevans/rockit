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


# Choose song
song = 'pop/tetris.mid'
s = converter.parse(song)
score_tempo = s.flat.getElementsByClass('MetronomeMark')[0]
score_instrument = s.flat.getElementsByClass('Instrument')[0]
# melody_instrument = instrument.Flute()
# harmony_instrument = instrument.SteelDrum()
melody_instrument = instrument.Piano()
harmony_instrument = instrument.Piano()

notes = s.flat.getElementsByClass('Note')
# notes = s.parts[0].flat.getElementsByClass('Note')
# chords = s.flat.getElementsByClass('Chord')
# chords = s.parts[0].flat.getElementsByClass('Chord')
chords = s.chordify().flat.getElementsByClass('Chord')

IDX2NOTE = list(set(note2str(n) for n in notes))
NOTE2IDX = {n: i for i, n in enumerate(IDX2NOTE)}
IDX2CHORD = list(set(chord2str(c) for c in chords))
CHORD2IDX = {c: i for i, c in enumerate(IDX2CHORD)}



N = len(IDX2NOTE)
C = len(IDX2CHORD)
CNNNtransition = np.zeros((C, N, N, N))
CCtransition = np.zeros((C, C))

ni = 0
prev_ni = 0
prev_prev_ni = 0
ci = 0
prev_ci = 0
notes = [(n.offset, 'note', note2str(n)) for n in notes]
notes = [notes[0]] + [next_n for n, next_n in zip(notes, notes[1:]) if next_n[2] != n[2]] # Ignore repeat of notes
chords = [(c.offset, 'chord', chord2str(c)) for c in chords]
chords = [chords[0]] + [next_c for c, next_c in zip(chords, chords[1:]) if next_c[2] != c[2]] # Ignore repeat of chords
combined = sorted(notes + chords)
for el in combined:
    if el[1] == 'note':
        prev_prev_ni, prev_ni = prev_ni, ni
        ni = NOTE2IDX[el[2]]
        CNNNtransition[ci, prev_prev_ni, prev_ni, ni] += 1
    elif el[1] == 'chord':
        prev_ci = ci
        ci = CHORD2IDX[el[2]]
        CCtransition[prev_ci, ci] += 1

# Turn into probabilities (TODO: Fix epsilon normalization)
CCtransition = normalize(CCtransition + 0.001, axis=1, norm='l1')
CNNNtransition = np.array([[normalize(y, axis=1, norm='l1') for y in x] for x in CNNNtransition + 0.001])


# Generate music!
melody = stream.Stream([melody_instrument])
harmony = stream.Stream([harmony_instrument])
# harmony = stream.Stream([instrument.SteelDrum(), harmony_tempo])
prev_ni = 0
ni = 0
ci = 0
for offset in range(100):
    if offset % 4 == 0: # Update chord
        probs = CCtransition[ci]
        ci = np.random.choice(range(len(probs)), p=probs)
        c = str2chord(IDX2CHORD[ci])
        c.duration = duration.Duration(4)
        try:
            c.inversion(random.choice([0, 1, 2]))
        except:
            pass
        harmony.insert(offset, c)
    # Update note
    probs = CNNNtransition[ci, prev_ni, ni]
    prev_ni = ni
    for _ in range(3):
        ni = np.random.choice(range(len(probs)), p=probs)
        n = str2note(IDX2NOTE[ni])
        if n.name in c.pitchNames:
            break
    else:
        n.style.color = 'pink'
    n.duration = duration.Duration(random.choice([0.5, 1, 1.5]))
    melody.insert(offset, n)
title = f"Inspired by {song.split('/')[-1]}"
s = stream.Stream([metadata.Metadata(title=title), score_tempo, melody, harmony])
s.write('midi', title.replace(' ', '_'))
s.write('mxl', title.replace(' ', '_') + '.mxl')
