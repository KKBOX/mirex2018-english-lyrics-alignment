import numpy as np
import scipy.io.wavfile
from scipy.interpolate import InterpolatedUnivariateSpline


def loadFile(filePath):
    with open(filePath, 'r') as fin:
        return fin.read().splitlines()


def dpOverMapM(stateProbMat, transProbMat):
    # transProbMat: row idx: from, column idx: to
    stateNum, frameNum = stateProbMat.shape
    dpTable = np.zeros((stateNum, frameNum), dtype=np.float64)
    pathTable = np.zeros((stateNum, frameNum), dtype=np.float64)
    dpPath = np.zeros((frameNum), dtype=np.int32)
    # --- Fill table
    dpTable[:, 0] = stateProbMat[:, 0]  # + np.log(1.0/stateNum)
    for f in range(1, frameNum):
        for s in range(stateNum):
            prob = dpTable[:, f-1] + transProbMat[:, s]
            maxVal = np.max(prob)
            maxIdx = np.argmax(prob)
            pathTable[s, f] = maxIdx
            dpTable[s, f] = stateProbMat[s, f] + maxVal
    # --- Backtrack
    maxIdx = np.argmax(dpTable[:, -1])
    dpPath[-1] = maxIdx
    for f in range(frameNum-2, -1, -1):
        dpPath[f] = pathTable[dpPath[f+1], f+1]
    return dpPath.astype('int32')


# =========================================================================


def waveResample(y, fs, fs2):
    # Following line causes memory error, use 'InterpolatedUnivariateSpline' instead
    # Ref: http://stackoverflow.com/questions/21435648/cubic-spline-memory-error
    # f = interp1d(np.arange(0, y.shape[0], dtype='float64')/fs, y, kind='cubic')
    f = InterpolatedUnivariateSpline(np.arange(0, y.shape[0], dtype='float64')/fs, y, k=3)
    return f(np.arange(0, y.shape[0]*1.0*fs2/fs, dtype='float64')/fs2)


def audioread(fileName, targetFs, toMono=True):
    fs, y = scipy.io.wavfile.read(fileName)
    if str(y.dtype) == 'int16':
        y = y.astype('float64') / 32768.0
    else:
        raise Exception('Unsupported wav file type: ' + str(y.dtype))
    if fs != targetFs:
        y = waveResample(y, fs, targetFs)
        fs = targetFs
    if toMono and len(y.shape) == 2:
        y = np.mean(y, axis=1)
    return y, fs

# =========================================================================

def getNeighborIdx(oriIdx, neighborOffset):
    idxNum = len(oriIdx)
    if neighborOffset < 0:
        return np.hstack([np.zeros((1, abs(neighborOffset)))[0]+oriIdx[0], oriIdx[0:(idxNum+neighborOffset)]]).astype('int32')
    elif neighborOffset == 0:
        return oriIdx.astype('int32')
    elif neighborOffset > 0:
        return np.hstack([oriIdx[neighborOffset:idxNum], np.zeros((1, abs(neighborOffset)))[0]+oriIdx[-1]]).astype('int32')


if __name__ == '__main__':
    oriIdx = np.array([3, 4, 5, 6, 7, 8, 9])
    print('XX:', oriIdx)
    print(' 0:', getNeighborIdx(oriIdx, 0))
    print('-1:', getNeighborIdx(oriIdx, -1))
