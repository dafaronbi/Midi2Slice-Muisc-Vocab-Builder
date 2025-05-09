# Midi2Slice-Muisc-Vocab-Builder
Python implementation of previous Matlab midi2slice code

musicWord2Vec
Adapted from paper “From Context to Concept: Exploring Semantic Relationships in Music with Word2Vec,” Neural Computing and Applications, 
Original Code in matlab [available here] (https://github.com/ChinghuaChuan/musicWord2Vec/tree/master)

Script build_music_vocab.py

This file converts midi files in a folder into sequences of music slices.
Example: 

`build_music_vocab.py <folder> <beat> <pc>`

<folder> is the directory where midi files are stored (all midi files in this directory will be converted to vocabulary slices) (default is the current folder)
<beat> is an integer of how many beats are used to comprise one musical slice (default is 1)
<pc> is an integer value (either 1 or anything else) indicating how pitches are represented in the music vocabulary. If pc = 1 then only 12 pitches are used (no difference for octabe) if another number is used then 88 pitches are used. (default is 1)

Output file 1: music_slice.txt --> this file is used for building word2vec embedding. Each row is a sequence of music slices from a midi file with a decimal value representing the binary coding of 12 pitch classes in each slice.
Output file 2: music_slice_file_list.txt --> this file lists the midi filename from which the music slices are generated.
Output file 3: vocab_slice_occurrence.txt --> the number of occurrences for each combination of 12 pitch classes.
To run build_music_vocab.m, be sure to have the following Matlab toolboxes:

