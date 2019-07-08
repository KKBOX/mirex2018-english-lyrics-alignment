import subprocess

import numpy as np
import scipy.signal
import scipy.fftpack

import myUtil


def windowSet(windowMethod, frameSize, trans=True):
    if windowMethod in ('none', ''):
        windowMethod = ''
        window = np.ones([frameSize])
    elif windowMethod == 'hamming':
        window = scipy.signal.hamming(frameSize)
    elif windowMethod == 'sin':
        window = scipy.sin(np.linspace(0, np.pi, frameSize))
    else:
        raise Exception('Unsupported windowMethod: ' + windowMethod)
    if trans:
        window = np.matrix(window).transpose()
    return windowMethod, window


def getCtxWindowFea(fea, ctxWindow, fileIdx, netType='mlp', printOpt=0):
    if fea.ndim > 2:
        return fea

    (oldDimNum, dataNum) = fea.shape
    if ctxWindow > 1 and ctxWindow % 2 == 1:
        neighborNum = ctxWindow // 2
        fea = np.vstack([np.zeros((oldDimNum*neighborNum, dataNum), dtype='float32'),
                         fea, np.zeros((oldDimNum*neighborNum, dataNum), dtype='float32')])
        uniqueFileIdx = np.unique(fileIdx)
        uniqueFileNum = len(uniqueFileIdx)
        _, dataNum = fea.shape
        for f in range(uniqueFileNum):
            base = 0
            fIdx = np.where(fileIdx == uniqueFileIdx[f])[0]
            for offset in range(-neighborNum, neighborNum+1):
                fea[(oldDimNum*base):(oldDimNum*(base+1)), fIdx] = fea[(oldDimNum*neighborNum):(oldDimNum*(neighborNum+1)), myUtil.getNeighborIdx(fIdx, offset)]
                base += 1
    elif ctxWindow == 1:
        pass
    else:
        raise ValueError('Invalid value of ctxWindow: {}'.format(ctxWindow))

    if netType == 'mlp':
        pass
    else:
        raise ValueError('Invalid value of netType: {}'.format(netType))

    return fea


def wavToFea(inputFileName, targetFs, frameSize, hopSize, windowParam, featype, feaParam):
    if not 'wav' in inputFileName:
        subprocess.call(['ffmpeg', '-v', '0', '-i', inputFileName, '-ar', str(targetFs), '-y', 'tmp.wav'])
        inputFileName = 'tmp.wav'

    if featype == 'mfcceda':
        import python_speech_features
        y, fs = myUtil.audioread(inputFileName, targetFs)
        mfcc_e = python_speech_features.base.mfcc(
            y, samplerate=targetFs, numcep=feaParam['numMfcc'],
            winlen=1.0*frameSize/targetFs, winstep=1.0*hopSize/targetFs,
            nfft=frameSize, winfunc=np.hamming)
        mfcc_e_d = python_speech_features.base.delta(mfcc_e, 1)
        mfcc_e_a = python_speech_features.base.delta(mfcc_e_d, 1)
        fea = np.hstack([mfcc_e, mfcc_e_d, mfcc_e_a]).T
        frameTime = (hopSize*np.arange(0, fea.shape[1], dtype='float64'))/fs
    else:
        raise Exception('Unsupported feature: ' + featype)
    return np.copy(fea), np.copy(frameTime)
