from scipy.io import wavfile
import numpy as np

def getFramesPower (fileName, fftSize):
    sample_rate, audio = wavfile.read(fileName)
    if len(np.shape(audio))==2:
        audio=audio[:,0]
    audio_duration = len(audio)/sample_rate

    window_length = 0.13 #in seconds
    window_hop = 0.065 #seperate successive windows in seconds

    #convert seconds to samples (integer)
    frame_length = int(window_length*sample_rate)#N nb d'échantillons par fenetre
    frame_hop = int(window_hop*sample_rate)#M

    #print(frame_length)
    #print(frame_hop)
    frames_overlap = frame_length - frame_hop #O = N-M
    audio_length = len(audio)

    """Si on note k le nb de fenêtres dans le signal, N la largeur d'une fenêtre et O l'overlap
    la longueur du signal vaut le nb de fenêtre - nb d'overlap + reste où reste<N-O
    donc la longueur du signal s'écrit kN - (k-1)O + reste = k(N-O) + O + reste
    On retrouve bien k avec (audio_length - O)//(N-O)
    """
    frames_number =  np.abs(audio_length - frames_overlap) // np.abs(frame_length - frames_overlap)
    reste = np.abs(audio_length - frames_overlap) % np.abs(frame_length - frames_overlap)
    #frames have the same number of samples
    if reste != 0:
        pad_length = int(frame_length - frames_overlap - reste) #pour compléter la dernière fenêtre, il faut rajouter N-O-reste
        pad = np.zeros(pad_length)
        padded_audio = np.append(audio,pad)
        frames_number +=1
    else :
        padded_audio = audio #padded_audio est une matrice colonne 

    index= np.zeros((frames_number, frame_length))
    for i in range (frames_number):
        for j in range (frame_length):
          index[i][j] = frame_hop*i+j
    frames = padded_audio[index.astype(np.int32, copy=False)] #réalise le framing, le false permet de directement cast indice et pas une copie

    """
    test
    print(np.shape(audio))
    print(np.shape(padded_audio))
    print(np.abs(len(padded_audio - frames_overlap) % np.abs(frame_length - frames_overlap))) #on verifie que le reste de padded audio est nul 
    print(np.shape(padded_audio)[0])
    print(audio)
    print(padded_audio[frame_length:2*frame_length])
    print(index)
    print(frames)
    print(frames[16])
    print(np.shape(padded_audio))
    print(217*1200)

    Pour comprendre le framing :
    mat = np.array([[2,3,4],[5,6,7]])
    mat[np.array([[0,1,0],[1,1,1]])]
    renvoie array([[[2, 3, 4],
            [5, 6, 7],
            [2, 3, 4]],

           [[5, 6, 7],
            [5, 6, 7],
            [5, 6, 7]]])
    or index est une matrice
    [[     0      1      2 ...   1197   1198   1199]
     [   720    721    722 ...   1917   1918   1919]
     [  1440   1441   1442 ...   2637   2638   2639]
     ...
     [154080 154081 154082 ... 155277 155278 155279]
     [154800 154801 154802 ... 155997 155998 155999]
     [155520 155521 155522 ... 156717 156718 156719]]
    chaque tableau du tableau permet de copier frame_length terme de padded_audio, en faisant attention aux overlap
    """
    frames = frames*np.hamming(frame_length) #pour éviter les distorsions dans le domaine fréquenciel quand on fera une FFT
    #Transformation de Fourier et énergie du spectre

    frames_magn = np.absolute(np.fft.rfft(frames, fftSize))
    frames_power = ((frames_magn)**2)/fftSize
    return frames_power,sample_rate
