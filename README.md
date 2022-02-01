# HHWWyy_DNN
### Authors: Joshuha Thomas-Wilsker
### Institutes: IHEP Beijing, CERN
Package used to train deep neural network for HH->WWyy analysis.

## Environment settings
Several non-standard libraries must be present in your python environment. To ensure they are present:

*On lxplus* you may need to do this via a virtualenv (example @ http://scikit-hep.org/root_numpy/start.html):

If its the first time:
```
export LCGENV_PATH=/cvmfs/sft.cern.ch/lcg/releases
/cvmfs/sft.cern.ch/lcg/releases/lcgenv/latest/lcgenv -p LCG_85swan2 --ignore Grid x86_64-slc6-gcc49-opt root_numpy > lcgenv.sh
echo 'export PATH=$HOME/.local/bin:$PATH' >> lcgenv.sh
```

Otherwise:
```
source lcgenv.sh
curl -O https://bootstrap.pypa.io/get-pip.py
python get-pip.py --user
pip install --user virtualenv
virtualenv <my_env>
source <my_env>/bin/activate
```

Check the following libraries are present:

- python 3.7
- shap
- keras
- tensorflow
- root
- root_numpy
- numpy

If they are missing:
```
pip install numpy
pip install root_numpy
```

*If you have root access* then setup a conda environment for python 3.7
```
conda create -n <env_title> python=3.7
```

Check the python version you are now using:
```
python --version
```

Check the aforementioned libraries are present (for which some you may need anaconda). If any packages (including those I may have missed from the list above) are missing the code,
you can add the package to the environment easily assuming it doesn't clash or require something
you haven't got in the environment setup:
```
conda install <new_library>
```

## Basic training
Running the code:
```
python train-BinaryDNN.py -t <0 or 1> -s <suffix_for_output_dir> -i <input_files_path>
```

The script 'train-BinaryDNN.py' performs several tasks:
- From 'input_variables.json' a list of input variables to use during training is compiled.
- With this information the 'input_files_path' will be used to locate two directories: 1 (Signal) containing the signal ntuples and the other containing the background samples (Bkgs).
- These files are used by the 'load_data' function to create a pandas dataframe.
- So you don't have to recreate the dataframe each time you want to run a new training using the same input variables, the dataframe is stored in the training output directory (in human readable format if you want to inspect it).
- The dataframe is split into a training and a testing sample (events are divided up randomly).
- If class/event weights are needed in order to overcome the class imbalance in the dataset, there are currently two methods to do this. The method used is defined in the hyper-parameter definition section. Search for the 'weights' variable. Other hyper-paramters can be hard coded here as well.
- If one chooses, the code can be used to perform a hyper-parameter scan using the '-p' argument.
- The code can be run in two mode:
    - If you want to perform the fit -t 1 = train new model from scratch.
    - If you just wanted to edit the plots (see plotting/plotter.py) -t 0 = make plots from the pre-trained model in training directory.
- The model is then fit.
- Several diagnostic plots are made by default: input variable correlations, input variable ranking, ROC curves, overfitting plots.
- The model along with a schematic diagram and .json containing a human readable version of the moel parameters is also saved.
- Diagnostic plots along with the model '.h5' and the dataframe will be stored in the output directory.

## The Plotting package

## Evaluating the networks performance
