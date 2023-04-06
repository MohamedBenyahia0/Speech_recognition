from os import listdir , path
from sys import argv

from glob import glob
import numpy as np
from calculateMFCC import getMFCC
from scipy.spatial import distance
from mydtw import dtw


def getCommandFromFile(File_Path):
    DATABASE_PATH = 'data_speech_val'
    commands = listdir(DATABASE_PATH)

    # We will use only a maximum of N occurences per word
    N = 10

    mfccs = []
    true_commands = []

    for l in commands:
        sounds = sorted(glob(path.join(DATABASE_PATH, l, '*.wav')))

        
        
        sounds = sounds[:N]
        
        
        for s in sounds:
            mfcc = getMFCC(s)
            mfccs.append(mfcc)
            true_commands.append(l)
            
    mfccs = np.array(mfccs,dtype=object)
    true_commands = np.array(true_commands)
    dmin, minCommand = np.inf, ""
    x = getMFCC(File_Path)

    currentCommand = true_commands[0]
    distancesToCurrentCommand = []
    for i in range(len(true_commands)):
        y= mfccs[i]
        if true_commands[i]==currentCommand:
            _,__,d= dtw(x,y,distance.euclidean)
            distancesToCurrentCommand.append(d)
        else :
            #medianDistanceToCurrentCommand =np.median(distancesToCurrentCommand) 
            medianDistanceToCurrentCommand =np.mean(sorted(distancesToCurrentCommand)[:3]) 
            if medianDistanceToCurrentCommand< dmin:
                dmin = medianDistanceToCurrentCommand
                minCommand = currentCommand
            
            distancesToCurrentCommand = []
            currentCommand= true_commands[i]
            

    return minCommand



def main():
    File_Path = argv[1]
    command = getCommandFromFile(File_Path)
    print(command)
if __name__ == "__main__":
    main()