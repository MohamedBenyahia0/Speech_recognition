import numpy as np
import matplotlib.pyplot as plt

def dtw(x,y,dist):
    nMFCC_X = np.shape(x)[0]
    nMFFC_Y = np.shape(y)[0]

    localDistanceMatrix = np.zeros((nMFCC_X,nMFFC_Y))

    for i in range(nMFCC_X):
        for j in range(nMFFC_Y):
            localDistanceMatrix[i,j]= dist(x[i],y[j])
    
    accumulatedDistanceMatrix = np.zeros((nMFCC_X+1,nMFFC_Y+1))
    traceback_mat = np.zeros((nMFCC_X, nMFFC_Y))

    

    for i in range(1,nMFCC_X+1):
        accumulatedDistanceMatrix[i][0]= np.inf 
    for j in range(1,nMFFC_Y+1):
        accumulatedDistanceMatrix[0][j]= np.inf

    for i in range(1,nMFCC_X+1):
        for j in range(1,nMFFC_Y+1):
            if(i==1 and j==1):
                accumulatedDistanceMatrix[i][j]= localDistanceMatrix[0][0]
            else:
                accumulatedDistanceMatrix[i][j]= min(accumulatedDistanceMatrix[i][j-1]+localDistanceMatrix[i-1][j-1],
                accumulatedDistanceMatrix[i-1][j]+localDistanceMatrix[i-1][j-1],
accumulatedDistanceMatrix[i-1][j-1]+localDistanceMatrix[i-1][j-1])
                traceback_mat[i-1,j-1]=np.argmin(np.array([accumulatedDistanceMatrix[i-1][j-1],accumulatedDistanceMatrix[i-1][j],accumulatedDistanceMatrix[i-1][j-1]]))
    
    # Strip infinity edges from cost_mat before returning
    accumulatedDistanceMatrix = accumulatedDistanceMatrix[1:, 1:]
    return (traceback_mat,accumulatedDistanceMatrix,accumulatedDistanceMatrix[-1][-1])

    

def calculatePath(traceback_mat,accumulatedDistanceMatrix):
    


    i = traceback_mat.shape[0]- 1
    j = traceback_mat.shape[1]- 1
    path = [(i, j)]
    while i > 0 and j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
            
        elif tb_type == 2:
            # Deletion
            j = j - 1
            
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    accumulatedDistanceMatrix = accumulatedDistanceMatrix[1:, 1:]
    return path[::-1]


def main():
    # DTW
    x = np.array([0, 0, 1, 1, 0, 0, -1, 0, 0, 0, 0])
    y = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0, -1, -0.5, 0, 0])
    traceback_mat, accumulatedDistanceMatrix,cost = dtw(x,y,lambda a,b : np.linalg.norm(a-b))
    path = calculatePath(traceback_mat,accumulatedDistanceMatrix)
   
    plt.figure(figsize=(6, 4))
    
    plt.title("Accumulated Distance Matrix")
    plt.imshow(accumulatedDistanceMatrix, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
    x_path, y_path = zip(*path)
    plt.plot(y_path, x_path)
    plt.show()

if __name__=="__main__":
    main()
    
    
    
    
    





