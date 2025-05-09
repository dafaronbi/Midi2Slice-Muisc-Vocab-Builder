import sys
import numpy as np
import os
import mido
import math

def binary_to_decimal(binary_array):
    out = []
    for sub_array in binary_array:
        out.append(int(sum([(2**i) * el for i,el in enumerate(sub_array)])))

    return np.array(out)

def midi2slice(file, beat, pc):
    midi = mido.MidiFile(file)

    #get tempo information
    tempos = []
    tempos_time = []

    for track in midi.tracks:
        cumtime=0
        for message in track:
            cumtime = cumtime + message.time
            if message.type == "set_tempo":
                tempos_time.append(cumtime)
                tempos.append(message.tempo)

    tempos = np.array(tempos)
    tempos_time = np.array(tempos_time)

    #fill note matrix from each midi message in each midi track   
    nmat = []
    current_tempo = 500000
    for track in midi.tracks:
        cumtime = midi.ticks_per_beat
        seconds = 0
        for message in track:

            #update time parameters
            cumtime = cumtime + message.time
            beat_move = cumtime / midi.ticks_per_beat
            seconds = seconds + message.time*1e-6*current_tempo/midi.ticks_per_beat

            #set new tempo
            if tempos.any():
                ind = np.argmax(tempos_time[cumtime >= tempos_time])
                current_tempo = tempos[ind]

            if message.type == "note_on" and message.velocity > 0:
                nmat.append([beat,-1,message.channel, message.note, message.velocity, seconds, -1])
            if (message.type == "note_on" and message.velocity == 0) or message.type == "note_off":
                idx = []
                for i,val in enumerate(nmat):
                    if val[1] == -1 and val[2] == message.channel and val[3] == message.note:
                        idx.append(i)

                if len(idx) == 0:
                    print('note-off with no matching note-on. skipping.')
                elif len(idx) > 1:
                    print('warning: found multiple note-on matches for note-off, taking first...')
                    idx = idx[0]
                else:
                    nmat[idx[0]][1] = beat_move - nmat[idx[0]][0]
                    nmat[idx[0]][6] = seconds - nmat[idx[0]][5]
                    

    nmat = np.array(nmat)

    #get rid of -1s
    midimat = nmat[nmat[:,6] != -1]

    #clean up the duration
    midimat[:, 0] = np.round(midimat[:,0]*1000)/1000
    midimat[:, 1] = np.round(midimat[:,1]*2000)/2000

    num_slices = math.floor(max((midimat[:, 0] + midimat[:, 1])/beat)) + 1

    #onset_mat: num_slices x num_pitches or num_pitch_classes,
    #88 keys: from A0 to C8
    highest_pitch = 108 #C8
    lowest_pitch = 21 #A0

    #12 pitch classes
    if pc == 1:
        onset_mat = np.zeros((num_slices,12))
    #all 88 keys
    else:
        onset_mat = np.zeros((num_slices,highest_pitch - lowest_pitch + 1))

    #for each pitch store beginning slices and end slice
    slice_mat = np.zeros((len(midimat),2))
    slice_mat[:, 0] = np.floor(midimat[:,0]/beat)
    slice_mat[:, 1] = np.floor(midimat[:,0]+midimat[:,1]/beat)

    #find pitches that start and end in same slice
    index = np.where(slice_mat[:, 0]- slice_mat[:, 1] == 0)[0]
    if index.any():
        value = np.ones((len(index), 1))
        if pc == 1:
            rowNos = np.int32(slice_mat[index, 0])
            colNos = np.int32(midimat[index, 3] % 12)
            onset_mat[rowNos, colNos] = 1
        else:
            index2 = np.where((midimat[index, 3] >= lowest_pitch) & (midimat[index, 3] <= highest_pitch))[0]
            if index2.any():
                rowNos = np.int32(slice_mat[index[index2], 0])
                colNos = np.int32(midimat[index[index2], 3] -lowest_pitch)
                onset_mat[rowNos, colNos] = 1

    #find pitches that start and end in multiple slices
    index = np.where(abs(slice_mat[:, 0]- slice_mat[:, 1])> 0)[0]
    if index.any():
        for idx in index:
            begin_slice = int(slice_mat[idx,0])
            end_slice = int(slice_mat[idx,1])

            value = 1

            #start slice
            if pc == 1:
                onset_mat[begin_slice, int(midimat[idx, 3] % 12)] = value
            else:
                if (midimat[idx, 3] >= lowest_pitch) & (midimat[idx, 3] <= highest_pitch):
                    onset_mat[begin_slice,int(midimat[idx, 3] -lowest_pitch)] = value
            
            #end slice
            if pc == 1:
                onset_mat[end_slice, int(midimat[idx, 3] % 12)] = value
            else:
                if (midimat[idx, 3] >= lowest_pitch) & (midimat[idx, 3] <= highest_pitch):
                    onset_mat[end_slice,int(midimat[idx, 3] -lowest_pitch) ] = value

            #in-between slices
            if end_slice - begin_slice > 0:
                slice_index = list(range(begin_slice+1, end_slice))
                
                if pc == 1:
                    onset_mat[slice_index, int(midimat[idx, 3] % 12)] = value
                else:
                    if (midimat[idx, 3] >= lowest_pitch) & (midimat[idx, 3] <= highest_pitch):
                        onset_mat[slice_index,int(midimat[idx, 3] -lowest_pitch)] = value
        
    return onset_mat
        



def main():

    #initialize variables from comqmand line. Default looks in the current folder every beat for 12 pitch classes
    #pc = 1: music slice is based on 12 pitch classes, pc = 0: pitchs from A0

    folder = "."
    beat = 1
    pc = 1

    if len(sys.argv) > 1:
        folder = sys.argv[1]

        if len(sys.argv) > 2:
            beat = int(sys.argv[2])

            if len(sys.argv) > 3:
                pc = int(sys.argv[3])

    fid = open("TESTmusic_slice.txt", "a")
    fid_log = open("TESTmusic_slice_file_list.txt", "a")
    fid_vocab = open("TESTvocab_slice_occurrence.txt", "a")

    files = [ file for file in os.listdir(folder) if file.endswith(".mid") ]

    #create vocabulary matrix based on
    if pc == 1:
        vocab_mat = np.zeros((2**13))
    else:
        vocab_mat = np.int32(np.zeros((1, 10)))

    for i,file in enumerate(files):
        onset_mat = midi2slice(file, beat, pc)
        
        if onset_mat.any():
            if pc == 1:
                dec_mat = binary_to_decimal(onset_mat)
                output = ""
                for j in range(len(dec_mat)):
                    vocab_mat[dec_mat[j]] = int(vocab_mat[dec_mat[j]] + 1)
                    output = output + str(dec_mat[j]) + " "
            else:
                dec_mat1 = binary_to_decimal(onset_mat[:, :3])
                dec_mat2 = binary_to_decimal(onset_mat[:, 3:15])
                dec_mat3 = binary_to_decimal(onset_mat[:, 15:27])
                dec_mat4 = binary_to_decimal(onset_mat[:, 27:39])
                dec_mat5 = binary_to_decimal(onset_mat[:, 39:51])
                dec_mat6 = binary_to_decimal(onset_mat[:, 51:63])
                dec_mat7 = binary_to_decimal(onset_mat[:, 63:75])
                dec_mat8 = binary_to_decimal(onset_mat[:, 75:87])
                dec_mat9 = binary_to_decimal(np.expand_dims(onset_mat[:, 87], -1))
                
                dec_mat = np.column_stack((dec_mat1, dec_mat2, dec_mat3, dec_mat4, dec_mat5, dec_mat6, dec_mat7, dec_mat8, dec_mat9))
                output = ""
                for j in range(len(dec_mat)):
                    nrows, ncols = vocab_mat[:,:9].shape
                    dtype={'names':['f{}'.format(i) for i in range(ncols)],
                        'formats':ncols * [vocab_mat.dtype]}
                    rows, index, _ = np.intersect1d(vocab_mat[:,:9].view(dtype), np.expand_dims(dec_mat[j,:9], 0).view(dtype),return_indices=True)
                    
                    if index.size > 0:
                        print("found existing word")
                        vocab_mat[index[0], 9] = vocab_mat[index[0], 9] + 1
                    else:
                        print("add new word")
                        vocab_mat = np.concatenate((vocab_mat, [np.concatenate((dec_mat[j], [1]))]))

                    if j == 0:
                        output = str(dec_mat[j][0]) + '-' + str(dec_mat[j][1]) + '-' + str(dec_mat[j][2]) + '-' + str(dec_mat[j][3]) + '-'  + str(dec_mat[j][4]) + '-'  + str(dec_mat[j][5]) + '-'  + str(dec_mat[j][6]) + '-'  + str(dec_mat[j][7]) + '-' + str(dec_mat[j][8])
                    else:
                        output = output + ', ' + str(dec_mat[j][0]) + '-' + str(dec_mat[j][1]) + '-' + str(dec_mat[j][2]) + '-' + str(dec_mat[j][3]) + '-'  + str(dec_mat[j][4]) + '-'  + str(dec_mat[j][5]) + '-'  + str(dec_mat[j][6]) + '-'  + str(dec_mat[j][7]) + '-' + str(dec_mat[j][8])

        fid_log.write(f"{file} \n")
        fid.write((f"{output} \n"))
        fid_vocab.write((f"{vocab_mat} \n"))
    
    fid.close()
    fid_log.close()
    fid_vocab.close()
    



if __name__ == "__main__":
    main()
