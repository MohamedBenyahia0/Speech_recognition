import os
import time
import glob
import numpy as np
from calculateMFCC import *

from mydtw import dtw


def getScore(DATABASE_PATH):
    labels = os.listdir(DATABASE_PATH)
    labels = [filename for filename in os.listdir(DATABASE_PATH) if os.path.isdir(os.path.join(DATABASE_PATH,filename))]

    # We will use only a maximum of N occurences per word
    N = 10

    mfccs = []
    true_labels = []

    for l in labels:
        sounds = sorted(glob.glob(os.path.join(DATABASE_PATH, l, '*.wav')))

        sounds = sounds[:N]
        
        for s in sounds:
            mfcc = getMFCC(s)
            mfccs.append(mfcc)
            true_labels.append(l)
            
    mfccs = np.array(mfccs,dtype=object)
    true_labels = np.array(true_labels)

    
    I_target = np.array([0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 
     17, 18, 19, 20,  22, 23, 24, 25, 26,  28, 29, 30, 31,  33, 34, 35, 36, 37,
      38,  40, 41, 42, 43, 44,  47, 48, 49, 50,  52, 53, 54, 55,  57, 58, 59, 60, 61,
       62, 63,  65, 66, 67, 68, 70, 71,  73, 74, 75, 76,  78, 79, 80,  82, 83, 84, 85,
        86, 87,  89, 90, 91, 92, 93, 94,  96, 97,  99])
    I_source= np.array([2, 7, 13, 16, 21, 27, 32, 39, 45,  46, 51, 56, 64, 69, 72, 77, 81, 88,  95, 98])
   

    

    score = 0.0

    for i in I_source:
        x = mfccs[i]


        dmin, minLabel = np.inf , ""
        distanceDictionary = {}
        for l in labels:
            distanceDictionary[l]=[]
        
        for j in I_target:
            y = mfccs[j]
            _,__,d = dtw(x,y,lambda a,b : np.linalg.norm(a-b))
            distanceDictionary[true_labels[j]].append(d)
        
        for l in labels:
            

            #medianDistanceToLabel = np.median(distanceDictionary[l])
            medianDistanceToLabel =np.mean(sorted(distanceDictionary[l])[:3])
            if medianDistanceToLabel < dmin:
                dmin = medianDistanceToLabel
                minLabel = l
        #print(true_labels[i] +" "+minLabel)
        score += 1.0 if (true_labels[i] == minLabel) else 0.0
           
        
    return score / len(I_source)



def main():
    DATABASE_PATH = 'data_speech_val'
    start = time.time()
    rec_rate = getScore(DATABASE_PATH)
    print('Recognition rate {}%'.format(100. * rec_rate))

    end = time.time()
    print(f"Time of execution of program is :{end-start} s")

if __name__ == "__main__":
    main()