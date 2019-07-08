import os
import argparse
import configparser

import numpy as np
import tensorflow as tf

import myAPI
import myUtil
import myFeaExt

np.set_printoptions(linewidth=150)


def main():
    args = get_args()
    run(args)


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='My tester for AME')
    parser.add_argument('modeldir', help='Model directory name')
    # --- Recognition mode param, no groundtruth
    parser.add_argument('--input', '-i', nargs='+', help='Input file names')
    parser.add_argument('--output', '-o', help='Output file name')
    # --- Others
    parser.add_argument('--modelpostfix', '-mpf', default='')
    parser.add_argument('--overwrite_output', action='store_true')

    args = parser.parse_args(argv)

    return args


def run(args):
    modelDir = args.modeldir
    # --- Recognition mode param
    inputFileNames = args.input
    outputFileName = args.output
    # --- Others
    modelPostfix = args.modelpostfix

    if inputFileNames is None or outputFileName is None:
        raise Exception('Missing parameter: input ot output files.')
    if os.path.isfile(outputFileName) and not args.overwrite_output:
        raise Exception('Output file already exist.')

    # --- Training target param, read from config file
    config = configparser.RawConfigParser()
    config.read(os.path.join(modelDir, 'config.ini'))
    # --- General
    netType = config.get('GENERAL', 'nettype')
    # --- MLP
    dimNum = config.getint('MLP', 'dimnum')
    hiddenLayers = list(map(int, config.get('MLP', 'hidden').replace(
        '[', '').replace(']', '').replace(',', '').split(' ')))
    outNum = config.getint('MLP', 'outNum')
    actiFuncName = config.get('MLP', 'activation')
    outActiFuncName = config.get('MLP', 'outact')
    try:
        useResNet = config.getboolean('MLP', 'useres')
    except Exception as e:
        useResNet = False
    # --- Target
    ctxWindow = config.getint('target', 'ctxwindow')

    # --- Fea config, also read from file
    featype = []
    feaNorm = []
    targetFs = []
    frameSize = []
    hopSize = []
    windowParam = []
    feaParam = []
    for dpt in myUtil.loadFile(os.path.join(modelDir, 'dsPostfix.ini')):
        config.read(os.path.join(modelDir, 'feaconfig_' + dpt + '.ini'))
        featype.append(config.get('feature', 'featype'))
        feaNorm.append(config.getboolean('feature', 'feanorm'))
        targetFs.append(config.getint('feature', 'fs'))
        frameSize.append(config.getint('feature', 'framesize'))
        hopSize.append(config.getint('feature', 'hopSize'))
        windowMethod = config.get('feature', 'window')
        wndMethod, wnd = myFeaExt.windowSet(windowMethod, frameSize[-1])
        windowParam.append({'method': wndMethod, 'window': wnd})
        feaParamDic = {}
        # --- FFT param
        try:
            feaParamDic['pad'] = config.getboolean('feature', 'pad')
        except Exception as e:
            feaParamDic['pad'] = 0
        # --- MEL param
        feaParamDic['numMel'] = config.getint('feature', 'nummel')
        try:
            feaParamDic['melUseLog'] = config.getint('feature', 'meluselog')
        except Exception as e:
            feaParamDic['melUseLog'] = 1
        # --- MFCC param
        try:
            feaParamDic['numMfcc'] = config.getint('feature', 'nummfcc')
        except Exception as e:
            feaParamDic['numMfcc'] = 20
        feaParam.append(feaParamDic)

    # --- Setup network
    phKeepProb = tf.placeholder(tf.float32)
    phUseRes = tf.placeholder(tf.bool)
    actiFunc = myAPI.setActiFunc(actiFuncName)
    outActiFunc = myAPI.setActiFunc(outActiFuncName)
    if netType == 'mlp':
        phInput = tf.placeholder(tf.float32, [None, dimNum])
        network = myAPI.MLP(dimNum, hiddenLayers, outNum, actiFunc, outActiFunc, 'u', 0)
        phComputed = network(phInput, phKeepProb, phUseRes)
    else:
        raise Exception('Unimplemented or unsupported netType: ' + netType)

    transProbMatVnv = np.load('trans_vnv.npy')

    # --- Restore session
    modelName = 'model{}.ckpt'.format(modelPostfix)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(modelDir, modelName))

    # --- Recognition
    for inputFileName in inputFileNames:
        print('Processing', inputFileName)
        fea = np.array([])
        for j in range(len(featype)):
            tmpFea, frameTime = myFeaExt.wavToFea(
                inputFileName, targetFs[j], frameSize[j], hopSize[j], windowParam[j], featype[j], feaParam[j])
            fea = np.vstack([fea, tmpFea]) if fea.size else tmpFea
        fea = myFeaExt.getCtxWindowFea(fea, ctxWindow, np.zeros(fea.shape[1]), netType=netType, printOpt=1)
        if fea.ndim == 2:
            fea = fea.T
        # --- recog
        feedDict = {
            phInput: fea,
            phKeepProb: 1.0,
            phUseRes: useResNet
        }
        recogMapFrame = sess.run(phComputed, feedDict).T

        maxIdxFrame = myUtil.dpOverMapM(recogMapFrame, np.log(transProbMatVnv))
        # --- Extend the vocal part (5 frames before)
        for i in range(1, len(maxIdxFrame)):
            if maxIdxFrame[i - 1] == 0 and maxIdxFrame[i] == 1:
                for j in range(max(0, i - 5), i):
                    maxIdxFrame[j] = 1
        # --- Output txt
        min_len = np.min([frameTime.shape[0], maxIdxFrame.shape[0], recogMapFrame.shape[1]])
        np.savetxt(outputFileName, np.vstack(
            [frameTime[:min_len], maxIdxFrame[:min_len], recogMapFrame[:, :min_len]]).T, fmt='%.3f %.4f %.8f %.8f')


if __name__ == "__main__":
    main()
