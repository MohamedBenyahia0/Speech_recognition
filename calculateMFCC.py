#Humans perceive frequency logarithmically
#Mel scale is a logarithmic scale
#Equal distances on the scale have same perceptual distance
#1000Hz = 1000 Mel
#m= 2595 * log(1+f/500)
#f= 700* (10^(m/2595)-1)


import math 
import numpy as np
import matplotlib.pyplot as plt
import scipy.fft
from Framing import *
#Algorithm :
#-Convert amplitude to DB
#-Convert frequencies to MEL scale:
#-1 Choose number of mel bands 
melPointsNum = 30
#-2     Construct mel filter banks
#-3 Apply mel filter banks to spectrogram





fftSize = 512#512 # usually a power of 2

#Construction of mel filter banks:

#-Converting lowest/highest frequency to Mel
def hertzToMel(freqs) :
    return 2595 * math.log10(1+freqs/700)
def melToHertz(mels):
    return 700 * (10**(mels/2595)-1)


# getMFFC takes as input pow_frames which represent the power spectrum 
def getMFCC(fileName):
    frames_power, sample_rate = getFramesPower(fileName,fftSize)
    lowestFreq = 0
    highestFreq = sample_rate/2
    lowestMel = hertzToMel(lowestFreq)
    highestMel = hertzToMel(highestFreq)
    # Creating bands equally spaced points
    melPoints = np.linspace(lowestMel,highestMel,melPointsNum+2)# +2 because we must take into consideration lowest_mel and highest_mel
    # Converting points back to hertz
    hertzPoints = melToHertz(melPoints)
    # Round to nearest frequency bin
    # #The bandwidth of the FFT is divided into bins , the number of which is 1/2 the FFT size
    # # the bin width defines the frequency resolution of the FFT
    # # the bin width equals the bandwith of the FFT divided by the number of bins 
    # # so the bin width equals sample_rate/ FFT_size
    FFTbins = np.floor(((fftSize + 1) /sample_rate )* hertzPoints )
    #Creating our triangular filterbanks
    # # The first filterbank will start at the first point, reach its peak at the second point , then return to zero at the third point
    # # the second filterbank will start at the second point, reach its peak at the third point, then return to zero at the fourth point
    # #etc...
    # # first, we initialize our filterbanks matrix
    # # number of lines is equal to melPointsNum
    # #number of columns is equal to fftSize/2 +1 which is the number of columns of frames_power
    filterbanks = np.zeros((melPointsNum, fftSize//2 +1))
    for m in range(1,melPointsNum):
        for k in range(int(FFTbins[m-1]),int(FFTbins[m])):
            filterbanks[m][k] = (k-int(FFTbins[m-1]))/(int(FFTbins[m])-int(FFTbins[m-1]))
        for k in range(int(FFTbins[m]),int(FFTbins[m+1])):
            filterbanks[m][k]= (int(FFTbins[m+1])-k)/(int(FFTbins[m+1])-int(FFTbins[m]))
    
    # we apply the mel filterbanks to the audio power spectrogram
    #melSpectogram[i][j] returns the power corresponding at the j bin frequency in the frame number i
    melSpectrogram = np.dot(frames_power,np.transpose(filterbanks))
    melSpectrogram = np.where(melSpectrogram == 0, np.finfo(float).eps, melSpectrogram)#stability replace zero with epsilon
    melSpectogram_log =  20 *np.log10(melSpectrogram)
    #we can apply Discrete Cosine Transform (DCT) to decorrelate the filter bank coefficients and yield a compressed representation of the filter banks
    # #for Automatic Speech Recognition (ASR), the resulting cepstral coefficients 2-13 are retained and the rest are discarded
    mfcc = scipy.fft.dct(melSpectogram_log,type=2,axis =1,norm='ortho') [ : , 2: 13]
    return mfcc





 



