# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#           train-DNN.py
#  Author: Joshuha Thomas-Wilsker
#  Institute of High Energy Physics
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Code to train deep neural network
# for HH->WWyy analysis.
import matplotlib.pyplot as plt
import numpy as np
import pickle
import shap
from array import array
import time
import pandas
import pandas as pd
import optparse, json, argparse, math
import ROOT
from ROOT import TTree
import tensorflow
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import log_loss
import os
from os import environ
from tensorflow import keras
#import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense,Activation,Flatten,Dropout,BatchNormalization
from tensorflow.keras.optimizers import Adam,Adamax,Nadam,Adadelta,Adagrad
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping
from plotting.plotter import plotter
#from root_numpy import tree2array

#from keras import backend as K
os.environ['KERAS_BACKEND'] = 'tensorflow'
seed = 7
np.random.seed(7)
rng = np.random.RandomState(31337)

def load_data_from_EOS(self, directory, mask='', prepend='root://eosuser.cern.ch'):
    eos_dir = '/eos/user/%s ' % (directory)
    eos_cmd = 'eos ' + prepend + ' ls ' + eos_dir
    print(eos_cmd)
    #out = commands.getoutput(eos_cmd)
    return

def load_data(inputPath,variables,criteria):
    # Load dataset to .csv format file
    my_cols_list=variables
    print('my_cols_list:\n',my_cols_list)
    data = pd.DataFrame(columns=my_cols_list)
    keys=['HH','bckg']
    for key in keys :
        print('key: ', key)
        if 'HH' in key:
            sampleNames=key
            subdir_name = 'Signal'
            fileNames = ['HHWWgg-SL-SM-NLO-2017']
            target=1
        else:
            sampleNames = key
            subdir_name = 'Bkgs'
            fileNames = [
            'DiPhotonJetsBox_MGG-80toInf_13TeV-Sherpa_Hadded',
            #'GJet_Pt-40toInf_DoubleEMEnriched_MGG-80toInf_TuneCP5_13TeV_Pythia8_Hadded',
            #'TTGJets_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8_Hadded',
            #'TTGG_0Jets_TuneCP5_13TeV_amcatnlo_madspin_pythia8_Hadded',
            #'TTJets_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8_Hadded',
            #'TTJets_HT-800to1200_TuneCP5_13TeV-madgraphMLM-pythia8_Hadded',
            #'TTJets_HT-1200to2500_TuneCP5_13TeV-madgraphMLM-pythia8_Hadded',
            #'TTJets_HT-2500toInf_TuneCP5_13TeV-madgraphMLM-pythia8_Hadded',
            #'W1JetsToLNu_LHEWpT_0-50_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',
            #'W1JetsToLNu_LHEWpT_50-150_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',
            'W1JetsToLNu_LHEWpT_150-250_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',
            'W1JetsToLNu_LHEWpT_250-400_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',
            'W1JetsToLNu_LHEWpT_400-inf_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',
            #'W2JetsToLNu_LHEWpT_0-50_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',
            #'W2JetsToLNu_LHEWpT_50-150_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',
            'W2JetsToLNu_LHEWpT_150-250_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',
            'W2JetsToLNu_LHEWpT_250-400_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',
            'W2JetsToLNu_LHEWpT_400-inf_TuneCP5_13TeV-amcnloFXFX-pythia8_Hadded',
            #'W3JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_Hadded',
            #'W4JetsToLNu_TuneCP5_13TeV-madgraphMLM-pythia8_Hadded',
            #'DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8_Hadded'
            #'ttHJetToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8_Hadded'
            ]
            target=0

        for filen in fileNames:
            if 'HHWWgg-SL-SM-NLO-2017' in filen:
                treename=['GluGluToHHTo2G2Qlnu_node_cHHH1_TuneCP5_PSWeights_13TeV_powheg_pythia8alesauva_2017_1_10_6_4_v0_RunIIFall17MiniAODv2_PU2017_12Apr2018_94X_mc2017_realistic_v14_v1_1c4bfc6d0b8215cc31448570160b99fdUSER']
                process_ID = 'HH'
            elif 'DiPhotonJetsBox_MGG' in filen:
                treename=['DiPhotonJetsBox_MGG_80toInf_13TeV_Sherpa',
                ]
                process_ID = 'DiPhoton'
            elif 'GJet_Pt-40toInf' in filen:
                treename=['GJet_Pt_40toInf_DoubleEMEnriched_MGG_80toInf_TuneCP5_13TeV_Pythia8',
                ]
                process_ID = 'GJet'
            elif 'DYJetsToLL_M-50_TuneCP5' in filen:
                treename=['DYJetsToLL_M_50_TuneCP5_13TeV_amcatnloFXFX_pythia8',
                ]
                process_ID = 'DY'
            elif 'TTGG' in filen:
                treename=['TTGG_0Jets_TuneCP5_13TeV_amcatnlo_madspin_pythia8',
                ]
                process_ID = 'TTGG'
            elif 'TTGJets' in filen:
                treename=['TTGJets_TuneCP5_13TeV_amcatnloFXFX_madspin_pythia8',
                ]
                process_ID = 'TTGJets'
            elif 'TTJets_HT-600to800' in filen:
                treename=['TTJets_HT_600to800_TuneCP5_13TeV_madgraphMLM_pythia8',
                ]
                process_ID = 'TTJets'
            elif 'TTJets_HT-800to1200' in filen:
                treename=['TTJets_HT_800to1200_TuneCP5_13TeV_madgraphMLM_pythia8',
                ]
                process_ID = 'TTJets'
            elif 'TTJets_HT-1200to2500' in filen:
                treename=['TTJets_HT_1200to2500_TuneCP5_13TeV_madgraphMLM_pythia8',
                ]
                process_ID = 'TTJets'
            elif 'TTJets_HT-2500toInf' in filen:
                treename=['TTJets_HT_2500toInf_TuneCP5_13TeV_madgraphMLM_pythia8',
                ]
                process_ID = 'TTJets'
            elif 'W1JetsToLNu_LHEWpT_0-50' in filen:
                treename=['W1JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W1JetsToLNu_LHEWpT_50-150' in filen:
                treename=['W1JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W1JetsToLNu_LHEWpT_150-250' in filen:
                treename=['W1JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W1JetsToLNu_LHEWpT_250-400' in filen:
                treename=['W1JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W1JetsToLNu_LHEWpT_400-inf' in filen:
                treename=['W1JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W2JetsToLNu_LHEWpT_0-50' in filen:
                treename=['W2JetsToLNu_LHEWpT_0_50_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W2JetsToLNu_LHEWpT_50-150' in filen:
                treename=['W2JetsToLNu_LHEWpT_50_150_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W2JetsToLNu_LHEWpT_150-250' in filen:
                treename=['W2JetsToLNu_LHEWpT_150_250_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W2JetsToLNu_LHEWpT_250-400' in filen:
                treename=['W2JetsToLNu_LHEWpT_250_400_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W2JetsToLNu_LHEWpT_400-inf' in filen:
                treename=['W2JetsToLNu_LHEWpT_400_inf_TuneCP5_13TeV_amcnloFXFX_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W3JetsToLNu' in filen:
                treename=['W3JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8',
                ]
                process_ID = 'WJets'
            elif 'W4JetsToLNu' in filen:
                treename=['W4JetsToLNu_TuneCP5_13TeV_madgraphMLM_pythia8',
                ]
                process_ID = 'WJets'
            elif 'ttHJetToGG' in filen:
                treename=['ttHJetToGG_M125_13TeV_amcatnloFXFX_madspin_pythia8',
                ]
                process_ID = 'ttH'

            fileName = os.path.join(subdir_name,filen)
            filename_fullpath = inputPath+"/"+fileName+".root"
            print("Input file: ", filename_fullpath)
            tfile = ROOT.TFile(filename_fullpath)
            for tname in treename:
                ch_0 = tfile.Get(tname)
                if ch_0 is not None :
                    # Create dataframe for ttree
                    chunk_arr = tree2array(tree=ch_0, branches=my_cols_list[:-5], selection=criteria)
                    #chunk_arr = tree2array(tree=ch_0, branches=my_cols_list[:-5], selection=criteria, start=0, stop=500)
                    # This dataframe will be a chunk of the final total dataframe used in training
                    chunk_df = pd.DataFrame(chunk_arr, columns=my_cols_list)
                    # Add values for the process defined columns.
                    # (i.e. the values that do not change for a given process).
                    chunk_df['key']=key
                    chunk_df['target']=target
                    chunk_df['weight']=chunk_df["weight"]
                    chunk_df['process_ID']=process_ID
                    chunk_df['classweight']=1.0
                    chunk_df['unweighted'] = 1.0
                    # Append this chunk to the 'total' dataframe
                    data = data.append(chunk_df, ignore_index=True)
                else:
                    print("TTree == None")
                ch_0.Delete()
            tfile.Close()
        if len(data) == 0 : continue

    return data

def load_trained_model(model_path):
    model = load_model(model_path, compile=False)
    return model

def baseline_model(num_variables,learn_rate=0.001):
    model = Sequential()
    model.add(Dense(32,input_dim=num_variables,kernel_initializer='glorot_normal',activation='relu'))
    #model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #model.compile(loss='binary_crossentropy',optimizer=Nadam(lr=learn_rate),metrics=['acc'])
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc'])
    return model

def gscv_model(learn_rate=0.001):
    model = Sequential()
    model.add(Dense(32,input_dim=29,kernel_initializer='glorot_normal',activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(4,activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    #model.compile(loss='binary_crossentropy',optimizer=Nadam(lr=learn_rate),metrics=['acc'])
    optimizer=Nadam(lr=learn_rate)
    model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['acc'])
    return model

def check_dir(dir):
    if not os.path.exists(dir):
        print('mkdir: ', dir)
        os.makedirs(dir)

def main():

    print('Using Keras version: ', keras.__version__)
    print('Using TF version: ', tensorflow.__version__)

    usage = 'usage: %prog [options]'
    parser = argparse.ArgumentParser(usage)
    parser.add_argument('-t', '--train_model', dest='train_model', help='Option to train model or simply make diagnostic plots (0=False, 1=True)', default=1, type=int)
    parser.add_argument('-i', '--inputs_file_path', dest='inputs_file_path', help='Path to directory containing directories \'Bkgs\' and \'Signal\' which contain background and signal ntuples respectively.', default='', type=str)
    parser.add_argument('-o', '--output', dest='output', help='Path to output directory (can be empty if new training, or existing directory with trained model etc)', default='', type=str)
    parser.add_argument('-p', '--para', dest='hyp_param_scan', help='Option to run hyper-parameter scan', default=0, type=int)
    args = parser.parse_args()
    do_model_fit = args.train_model
    output = args.output

    # Set model hyper-parameters
    weights='BalanceYields'# 'BalanceYields' or 'BalanceNonWeighted'
    optimizer = 'Nadam'
    validation_split=0.1
    # hyper-parameter scan results
    if weights == 'BalanceNonWeighted':
        learn_rate = 0.0005
        epochs = 200
        batch_size = 200
    if weights == 'BalanceYields':
        learn_rate = 0.0001
        epochs = 200
        batch_size = 400

    # Create instance of the input files directory
    inputs_file_path = args.inputs_file_path
    hyp_param_scan = args.hyp_param_scan

    # Create instance of output directory where all results are saved.
    output_directory = output
    check_dir(output_directory)

    # Create plots subdirectory
    plots_dir = os.path.join(output_directory,'plots/')

    # Fetch list of input variables
    #input_var_jsonFile = open('input_variables.json','r')
    #input_var_jsonFile = open('input_variables_new.json','r')
    intput_json_name = os.path.join(output_directory,'input_variables.json')
    input_var_jsonFile = open(intput_json_name,'r')
    variable_list = json.load(input_var_jsonFile,encoding="utf-8").items()

    # Set hyperparam_file
    hyperparam_file = os.path.join(output_directory,'additional_model_hyper_params.txt')
    additional_hyperparams = open(hyperparam_file,'w')
    additional_hyperparams.write("optimizer: "+optimizer+"\n")
    additional_hyperparams.write("learn_rate: "+str(learn_rate)+"\n")
    additional_hyperparams.write("epochs: "+str(epochs)+"\n")
    additional_hyperparams.write("validation_split: "+str(validation_split)+"\n")
    additional_hyperparams.write("weights: "+weights+"\n")

    # Define training domain
    selection_criteria = '( ((Leading_Photon_pt/CMS_hgg_mass) > 0.35) && ((Subleading_Photon_pt/CMS_hgg_mass) > 0.25) && passbVeto==1 && ExOneLep==1 && N_goodJets>=1)'
    #selection_criteria = ''

    # Create list of headers for dataset .csv
    column_headers = []
    print("original variable_list:\n",variable_list)
    for key,var in variable_list:
        column_headers.append(key)
    column_headers.append('weight')
    column_headers.append('weight_NLO_SM')
    column_headers.append('unweighted')
    column_headers.append('target')
    column_headers.append('key')
    column_headers.append('classweight')
    column_headers.append('process_ID')

    # Load ttree into .csv including all variables listed in column_headers
    newdataframename = '%s/output_dataframe.csv' %(output_directory)
    print('<train-DNN> Looking in output path %s for pre-existing output_dataframe.csv' % newdataframename)
    if os.path.isfile(newdataframename):
        print('<train-DNN> Dataframe already exists! Loading data .csv from: %s . . . . ' % (newdataframename))
        data = pandas.read_csv(newdataframename)
        print('<train-DNN> Dataframe column headers:\n' , data.columns.values.tolist())
        column_headers = data.columns.values.tolist()
    else:
        print('<train-DNN> Creating new data .csv @: %s . . . . ' % (inputs_file_path))
        data = load_data(inputs_file_path,column_headers,selection_criteria)
        # Change sentinal value to speed up training.
        data = data.mask(data<-25., -9.)
        #data = data.replace(to_replace=-999.000000,value=-9.0)
        data.to_csv(newdataframename, index=False)
        data = pandas.read_csv(newdataframename)

    print('<main> data columns: ', (data.columns.values.tolist()))
    n = len(data)
    nHH = len(data.iloc[data.target.values == 1])
    nbckg = len(data.iloc[data.target.values == 0])
    print("Total (train+validation) length of HH = %i, bckg = %i" % (nHH, nbckg))

    # Make instance of plotter tool
    Plotter = plotter()
    # Create statistically independant training/testing data
    traindataset, valdataset = train_test_split(data, test_size=0.1)
    #valdataset.to_csv(os.path.join(output_directory,'valid_dataset.csv'), index=False)

    print('<train-DNN> Training dataset shape: ', traindataset.shape)
    print('<train-DNN> Validation dataset shape: ', valdataset.shape)
    #print('traindataset: \n', traindataset.info())

    # Event weights
    weights_for_HH = traindataset.loc[traindataset['process_ID']=='HH', 'weight']
    weights_for_HH_NLO = traindataset.loc[traindataset['process_ID']=='HH', 'weight_NLO_SM']
    weights_for_Hgg = traindataset.loc[traindataset['process_ID']=='Hgg', 'weight']
    weights_for_DiPhoton = traindataset.loc[traindataset['process_ID']=='DiPhoton', 'weight']
    weights_for_GJet = traindataset.loc[traindataset['process_ID']=='GJet', 'weight']
    weights_for_QCD = traindataset.loc[traindataset['process_ID']=='QCD', 'weight']
    weights_for_DY = traindataset.loc[traindataset['process_ID']=='DY', 'weight']
    weights_for_TTGsJets = traindataset.loc[traindataset['process_ID']=='TTGsJets', 'weight']
    weights_for_WGsJets = traindataset.loc[traindataset['process_ID']=='WGsJets', 'weight']
    weights_for_WW = traindataset.loc[traindataset['process_ID']=='WW', 'weight']

    HHsum_weighted= sum(weights_for_HH)
    Hggsum_weighted= sum(weights_for_Hgg)
    DiPhotonsum_weighted= sum(weights_for_DiPhoton)
    GJetsum_weighted= sum(weights_for_GJet)
    QCDsum_weighted= sum(weights_for_QCD)
    DYsum_weighted= sum(weights_for_DY)
    TTGsJetssum_weighted= sum(weights_for_TTGsJets)
    WGsJetssum_weighted= sum(weights_for_WGsJets)
    WWsum_weighted= sum(weights_for_WW)
    bckgsum_weighted = Hggsum_weighted + DiPhotonsum_weighted + GJetsum_weighted + QCDsum_weighted + DYsum_weighted + TTGsJetssum_weighted + WGsJetssum_weighted + WWsum_weighted

    nevents_for_HH = traindataset.loc[traindataset['process_ID']=='HH', 'unweighted']
    nevents_for_Hgg = traindataset.loc[traindataset['process_ID']=='Hgg', 'unweighted']
    nevents_for_DiPhoton = traindataset.loc[traindataset['process_ID']=='DiPhoton', 'unweighted']
    nevents_for_GJet = traindataset.loc[traindataset['process_ID']=='GJet', 'unweighted']
    nevents_for_QCD = traindataset.loc[traindataset['process_ID']=='QCD', 'unweighted']
    nevents_for_DY = traindataset.loc[traindataset['process_ID']=='DY', 'unweighted']
    nevents_for_TTGsJets = traindataset.loc[traindataset['process_ID']=='TTGsJets', 'unweighted']
    nevents_for_WGsJets = traindataset.loc[traindataset['process_ID']=='WGsJets', 'unweighted']
    nevents_for_WW = traindataset.loc[traindataset['process_ID']=='WW', 'unweighted']

    HHsum_unweighted= sum(nevents_for_HH)
    Hggsum_unweighted= sum(nevents_for_Hgg)
    DiPhotonsum_unweighted= sum(nevents_for_DiPhoton)
    GJetsum_unweighted= sum(nevents_for_GJet)
    QCDsum_unweighted= sum(nevents_for_QCD)
    DYsum_unweighted= sum(nevents_for_DY)
    TTGsJetssum_unweighted= sum(nevents_for_TTGsJets)
    WGsJetssum_unweighted= sum(nevents_for_WGsJets)
    WWsum_unweighted= sum(nevents_for_WW)
    bckgsum_unweighted = Hggsum_unweighted + DiPhotonsum_unweighted + GJetsum_unweighted + QCDsum_unweighted + DYsum_unweighted + TTGsJetssum_unweighted + WGsJetssum_unweighted + WWsum_unweighted

    HHsum_weighted = 2*HHsum_weighted
    HHsum_unweighted = 2*HHsum_unweighted

    if weights=='BalanceYields':
        traindataset.loc[traindataset['process_ID']=='HH', ['classweight']] = HHsum_unweighted/HHsum_weighted
        traindataset.loc[traindataset['process_ID']=='Hgg', ['classweight']] = (HHsum_unweighted/bckgsum_weighted)
        traindataset.loc[traindataset['process_ID']=='DiPhoton', ['classweight']] = (HHsum_unweighted/bckgsum_weighted)
        traindataset.loc[traindataset['process_ID']=='GJet', ['classweight']] = (HHsum_unweighted/bckgsum_weighted)
        traindataset.loc[traindataset['process_ID']=='QCD', ['classweight']] = (HHsum_unweighted/bckgsum_weighted)
        traindataset.loc[traindataset['process_ID']=='DY', ['classweight']] = (HHsum_unweighted/bckgsum_weighted)
        traindataset.loc[traindataset['process_ID']=='TTGsJets', ['classweight']] = (HHsum_unweighted/bckgsum_weighted)
        traindataset.loc[traindataset['process_ID']=='WGsJets', ['classweight']] = (HHsum_unweighted/bckgsum_weighted)
        traindataset.loc[traindataset['process_ID']=='WW', ['classweight']] = (HHsum_unweighted/bckgsum_weighted)
    if weights=='BalanceNonWeighted':
        traindataset.loc[traindataset['process_ID']=='HH', ['classweight']] = 1.
        traindataset.loc[traindataset['process_ID']=='Hgg', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='DiPhoton', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='GJet', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='QCD', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='DY', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='TTGsJets', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='WGsJets', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)
        traindataset.loc[traindataset['process_ID']=='WW', ['classweight']] = (HHsum_unweighted/bckgsum_unweighted)

    # Remove column headers that aren't input variables.
    # Check the structure of your dataframe before editing.
    #training_columns = column_headers[:-6]
    #training_columns = column_headers[:-7]
    training_columns = column_headers[:-8]
    print('<train-DNN> Training features: ', training_columns)

    # Store column order for training.
    # Needed when sharing model with others.
    column_order_txt = '%s/column_order.txt' % (output_directory)
    column_order_file = open(column_order_txt, "wb")
    for tc_i in training_columns:
        line = tc_i+"\n"
        pickle.dump(str(line), column_order_file)

    num_variables = len(training_columns)

    # Extract training and testing data
    X_train = traindataset[training_columns].values
    X_test = valdataset[training_columns].values

    # Extract labels data
    Y_train = traindataset['target'].values
    Y_test = valdataset['target'].values

    # Event weights if wanted
    train_weights = traindataset['weight'].values*traindataset['weight_NLO_SM'].values
    test_weights = valdataset['weight'].values*valdataset['weight_NLO_SM'].values
    #train_weights = traindataset['weight'].values
    #test_weights = valdataset['weight'].values

    if do_model_fit == 1:
        # Weights applied during training.
        if weights=='BalanceYields':
            trainingweights = traindataset['classweight']*traindataset['weight']*traindataset.loc[:,'weight_NLO_SM']
        if weights=='BalanceNonWeighted':
            trainingweights = traindataset['classweight']
        trainingweights = np.array(trainingweights)

        ## Input Variable Correlation plot
        # Create dataframe containing input features only (for correlation matrix)
        #train_df = data.iloc[:traindataset.shape[0]]
        #correlation_plot_file_name = 'correlation_plot.pdf'
        #Plotter.correlation_matrix(train_df)
        #Plotter.save_plots(dir=plots_dir, filename=correlation_plot_file_name)

        print('<train-BinaryDNN> Training new model . . . . ')
        histories = []
        labels = []

        if hyp_param_scan == 1:
            print('Begin at local time: ', time.localtime())
            hyp_param_scan_name = 'hyp_param_scan_results.txt'
            hyp_param_scan_results = open(hyp_param_scan_name,'a')
            time_str = str(time.localtime())+'\n'
            hyp_param_scan_results.write(time_str)
            hyp_param_scan_results.write(weights)
            learn_rates=[0.00001, 0.0001]
            epochs = [150,200]
            batch_size = [400,500]
            param_grid = dict(learn_rate=learn_rates,epochs=epochs,batch_size=batch_size)
            model = KerasClassifier(build_fn=gscv_model,verbose=0)
            grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
            grid_result = grid.fit(X_train,Y_train,shuffle=True,sample_weight=trainingweights)
            print("Best score: %f , best params: %s" % (grid_result.best_score_,grid_result.best_params_))
            hyp_param_scan_results.write("Best score: %f , best params: %s\n" %(grid_result.best_score_,grid_result.best_params_))
            means = grid_result.cv_results_['mean_test_score']
            stds = grid_result.cv_results_['std_test_score']
            params = grid_result.cv_results_['params']
            for mean, stdev, param in zip(means, stds, params):
                print("Mean (stdev) test score: %f (%f) with parameters: %r" % (mean,stdev,param))
                hyp_param_scan_results.write("Mean (stdev) test score: %f (%f) with parameters: %r\n" % (mean,stdev,param))
            exit()
        else:
            # Define model for analysis
            early_stopping_monitor = EarlyStopping(patience=30, monitor='val_loss', verbose=1)
            model = baseline_model(num_variables, learn_rate=learn_rate)

            # Fit the model
            # Batch size = examples before updating weights (larger = faster training)
            # Epoch = One pass over data (useful for periodic logging and evaluation)
            history = model.fit(X_train,Y_train,validation_split=validation_split,epochs=epochs,batch_size=batch_size,verbose=1,shuffle=True,sample_weight=trainingweights,callbacks=[early_stopping_monitor])
            histories.append(history)
            labels.append(optimizer)
            # Make plot of loss function evolution
            Plotter.plot_training_progress_acc(histories, labels)
            acc_progress_filename = 'DNN_acc_wrt_epoch.png'
            Plotter.save_plots(dir=plots_dir, filename=acc_progress_filename)
            # Store model in file
            model_output_name = os.path.join(output_directory,'model.h5')
            model.save(model_output_name)
            weights_output_name = os.path.join(output_directory,'model_weights.h5')
            model.save_weights(weights_output_name)
            model_json = model.to_json()
            model_json_name = os.path.join(output_directory,'model_serialised.json')
            with open(model_json_name,'w') as json_file:
                json_file.write(model_json)
    else:
        model_name = os.path.join(output_directory,'model.h5')
        print('Fetching model pre-trained: ', model_name)
        model = load_trained_model(model_name)

    #model.summary()
    #model_schematic_name = os.path.join(output_directory,'model_schematic.png')
    #plot_model(model, to_file=model_schematic_name, show_shapes=True, show_layer_names=True)

    # Node probabilities for training sample events
    #result_probs = model.predict(np.array(X_train))
    #print('result_probs: ', result_probs)
    # Node probabilities for testing sample events
    #result_probs_test = model.predict(np.array(X_test))

    # Initialise output directory.
    Plotter.plots_directory = plots_dir
    Plotter.output_directory = output_directory

    # Make overfitting plots of output nodes
    #Plotter.binary_overfitting(model, Y_train, Y_test, result_probs, result_probs_test, plots_dir, train_weights, test_weights)

    # Input variable ranking (approximation via Shap values)
    # Explain the DNN predictions

    #simple_model_only_first_output = tensorflow.keras.Model(
    #    inputs=model.inputs,
    #    outputs=model.layers[-1].output[0],  # specifying a single output for shap usage
    #)
    #simple_model_only_first_output.summary()

    #e = shap.DeepExplainer(simple_model_only_first_output, X_train[:30])
    #de = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output[0]), X_train[:400, ])
    #deep_shap_values = de.shap_values(X_test[:30])

    #Plotter.plot_dot(title="DeepExplainer_sigmoid_y0", x=X_test[:400, ], shap_values=shap_values, column_headers=column_headers)
    #Plotter.plot_dot_bar(title="DeepExplainer_Bar_sigmoid_y0", x=X_test[:400,], shap_values=shap_values, column_headers=column_headers)

    N_shapley_examples = 600
    shap_values_file = os.path.join(output_directory,"shapley_values.npy")
    if os.path.isfile(shap_values_file):
        print("Using Shapley file: ",shap_values_file )
        shap_values = np.load(shap_values_file)
        for output_nodes in range(3):
            Plotter.plot_dot(title="KernalExplainer_sigmoid_y0_node"+str(output_nodes), x=X_train[:N_shapley_examples], shap_values=shap_values[output_nodes], column_headers=column_headers)
            Plotter.plot_dot_bar(title="KernalExplainer_Bar_sigmoid_y0_node"+str(output_nodes), x=X_train[:N_shapley_examples], shap_values=shap_values[output_nodes], column_headers=column_headers)
    else:
        print("Creating new Shapley file: ",shap_values_file )
        e = shap.KernelExplainer(model.predict_proba, X_train[:N_shapley_examples])
        shap_values = e.shap_values(X_train[:N_shapley_examples])
        np.save(os.path.join(output_directory,"shapley_values"),shap_values)
        for output_nodes in range(3):
            Plotter.plot_dot(title="KernalExplainer_sigmoid_y0_node"+str(output_nodes), x=X_train[:N_shapley_examples], shap_values=shap_values[output_nodes], column_headers=column_headers)
            Plotter.plot_dot_bar(title="KernalExplainer_Bar_sigmoid_y0_node"+str(output_nodes), x=X_train[:N_shapley_examples], shap_values=shap_values[output_nodes], column_headers=column_headers)

    # ROC curve
    #Plotter.ROC_sklearn(Y_train, result_probs, Y_test, result_probs_test, 1 , 'BinaryClassifierROC',train_weights, test_weights)

main()
