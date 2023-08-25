This repository should contain the code needed to replicate the main claims in the paper.

There are two primary experiments - one for the website fingerprinting setting and one for the flow correlation setting. 

The experiments were done on Ubuntu 20.04 using python 3.8 and an RTX 3090 (CUDA version 11.7). Other requirements can be found in requirements.txt.

If you'd like to make changes to script behavior (such as padding volume, choice of dataset, or CUDA device), then you'll find the relevant python variables near the file headers. 


### To run the website fingerprinting experiment:

To simplify the experiment, we provide wf_experiment.sh, which will simulate WF-DeTorrent on the BigEnough dataset. Note that the dataset is formatted such that the filenames include the website and instance number (e.g. website 10-50 is the 50th instance of the 10th website). Each file contains the metadata for a trace, where the left column includes the timestamps and the right column includes the cell size (where negative sizes represent download cells, and positive sizes represent upload cells).

First, wf_preprocessing.py will preprocess the data into .npy files and store them in wf_preprocessed_data. Note that we split the dataset 5 ways (similar to a 5-fold cross-validation). This lets us train a model on one portion of the dataset and defend the other portion (thus avoiding data leakage).

Next, the script runs wf_defense.py. This trains WF-DeTorrent on one set of dataset partitions and simulates the defense on the remaining partition (determined by command line argument shown in wf_experiment.sh). The defended traces will be output to wf_defense_output.

Once the defended dataset is available, use 'https://github.com/msrocean/Tik_Tok' to download and implement the Tik-Tok attack. We used a command similar to 'python3 cw_attack.py -a 1 -t /path/to/wf_defense_output' to run the Tik-Tok attack. Note that we changed the input size from 5000 to 10,000 to account for the larger traces in BE. We also increased the training 'patience' value to 10, given that Tik-Tok sometimes stops training prematurely. 

To run the Deep Fingerprinting attack, you can either download the relevant repository (https://github.com/deep-fingerprinting/df) or you can re-run the above command with '-a 0' to prevent timing information from being used (as the Tik-Tok attack is essentially just the Deep Fingerprinting attack with timing information). 

### To run the flow correlation experiment:

We provide fc_experiment.sh, which will simulate FC-DeTorrent on the DCF dataset. You'll have to unzip the attached inflows and outflows and add them to the 'dcf_dataset' folder (such that the inflows are contained in dcf_dataset/inflow/ and the outflows in dcf_dataset/outflow/).

First, fc_experiment.sh runs fc_preprocessing.py to preprocess the DCF dataset and split it into 5 folds for later defense. It will be saved into fc_data_cv. The DCF dataset is split into inflows and outflows, where each file contains one flow. Within each file, the first column contains the timestamps and the second column has the packet sizes (where positive sizes represent upload traffic and negative sizes represent download traffic). 

Then, the script runs fc_defense.py to train a set of 5 FC-DeTorrent models, where each is simulated on one partition of the dataset and outputs the defended traces, which will be saved to fc_defense_output. 

To test the DCF attack on the defended data, copy the defended output data into the original DCF dataset 'inflow' folder so that it replaces the original inflow data (as FC-DeTorrent just defends the inflow traffic) and run the DCF attack on the resulting data. The DCF attack code is available here: https://github.com/traffic-analysis/deepcoffea.


