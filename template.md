# Artifact Appendix

Paper title: DeTorrent: An Adversarial Padding-only Traffic Analysis Defense

Artifacts HotCRP Id: This paper is #114 at the PoPETS 2024.1 HotCRP and submission #16 for the artifacts HotCRP.

Requested Badge: Available (I've provided datasets, code, and instructions to allow for other researchers to replicate the paper's main claims and build off the work. However, running them requires specialized hardware, and it's difficult to provide a portable VM that could be guaranteed to work with the reviewer's available GPU. So, I don't necessarily expect the reviewers to fully replicate the results). 

## Description
This artifact simulates the DeTorrent defense on the BigEnough dataset (in the website fingerprinting setting) and on the DeepCoFFEA dataset (in the flow correlation setting). This substantiates the paper's claims about the DeTorrent traffic analysis defense being able to prevent both website fingerprinting and flow correlation attacks. The output of the code and attached data is a defended dataset, which can then be used as input to traffic analysis attacks such as Tik-Tok and DeepCoFFEA. 

### Security/Privacy Issues and Ethical Concerns
There are no relevant security or privacy concerns for the reviewer. 

## Basic Requirements

### Hardware Requirements

The experiments were done using an RTX 3090 (CUDA version 11.7). Both experiments used the full 24GB of VRAM with the default batch sizes.

GPUs can be bought or rented from a cloud provider, and many Nvidia GPUs produced in the past several years should work. However, note that the batch size in wf_defense.py and fc_defense.py may have to be reduced to account for the GPU's available VRAM. 

### Software Requirements
The experiments were done on Ubuntu 20.04 using python 3.8 and CUDA version 11.7. A requirements.txt file is included to specify the necessary versions of pytorch, numpy, and other python packages. These requirements should all be publicly available. 

### Estimated Time and Storage Consumption
 The website fingerprinting experiment takes about four hours to run on this hardware, with most of the time spent training the defense generator. The flow correlation experiment completes in about an hour.

The unzipped datasets use about 5GB of disk space total, though about 20GB may be needed for intermediate files. Up to 24GB of GPU VRAM may be required.

## Environment
You may set up your environment with `pip3 install -r requirements.txt`. You'll also have to make sure that you've installed a compatible version of CUDA to work with your GPU.

### Accessibility

The artifacts can be found at https://github.com/jkhollandjr/PETS_DeTorrent. The datasets are hosted in the repository using github large file storage. The most recent commit should be used by the PETS reviewer.

### Set up the environment

See the README for detailed instructions about the experiments and what commands to run. 

```bash
git clone https://github.com/jkhollandjr/PETS_DeTorrent.git
pip3 install -r requirements.txt
```

### Testing the Environment

If the environment is not correct, then python should provide errors about missing packages or incompatibilities. This will most likely happen when running wf_defense.py or fc_defense.py.

## Artifact Evaluation

### Main Results and Claims

#### Main Result 1: Name

The first claim is that DeTorrent is a strong website fingerprinting defense. To demonstrate this, we train the DeTorrent defense on the BigEnough dataset and create a defense generator that can be used to simulate DeTorrent. Then, we output the defended traces, which can be used as input to website fingerprinting attacks (with the expectation that attack performance will be significantly reduced). See section 5.2 in the paper for details and section 6.1 for results.

#### Main Result 2: Name

The second claim is that DeTorrent can be used to defend against flow correlation. Accordingly, we train the defense on the attached DeepCoFFEA dataset and output a defended version. This version can then be used by the DeepCoFFEA attack to see the reduction in attack performance. See section 5.3 in the paper for details and section 7 for results.
...

### Experiments
List each experiment the reviewer has to execute. Describe:
 - How to execute it in detailed steps.
 - What the expected result is.
 - How long it takes and how much space it consumes on disk. (approximately)
 - Which claim and results does it support, and how.


#### Experiment 1: Name
```bash
unzip be_dataset.zip
./wf_experiment.sh
```

The first command simply unzips the dataset, and the second command starts the data preprocessing, model training, and defended trace output process. This will take several hours to run on a 3090 GPU and use up to ~24GB VRAM along with 20GB of disk space. This script outputs the defended dataset to support the claim that DeTorrent is a strong website fingerprinting defense. 

To see the Tik-Tok attack performance reduction, clone the Tik-Tok repo (https://github.com/msrocean/Tik_Tok) and follow the contained instructions. The most relevant file there is cw_attack.py and the directional timing representation should be used. The input directory will be the newly generated wf_defense_output directory.

For more details, please see README.md.

#### Experiment 2: Name
```bash
unzip dcf_dataset.zip
./fc_experiment.sh
```

Similarly, the first command unzips the dataset while the second command starts the data preprocessing, model training, and defended trace output processing. This will take about an hour to run and will consume up to 24GB of GPU VRAM and 20GB of disk space. This experiment outputs the defended DeepCoFFEA dataset in order to demonstrate DeTorrent's ability to prevent flow correlation attacks.

To see the reduction in flow correlation attack performance, clone the DeepCoFFEA repository and follow the included instructions, using the newly created defended output (fc_defense_output) as input.

For more details, please see README.md.

## Limitations
Note that we have not included the Tik-Tok and DeepCoFFEA source code in this repository, as this would be redundant and overly complicate the software dependencies. However, see the file README.md for instructions about where to find these attacks and how to run them. The output of the experiments described in this document can be used as input to these attacks. 

## Notes on Reusability
The included source code could be used to design and train traffic analysis defenses for eventual deployment. Also, even if the specific DeTorrent defense isn't used, the GAN-like framework of having a defense generator iteratively train against a discriminator could be used to design better traffic analysis defenses. 

Changing code parameters, such as bandwidth overhead and trace representation length, will also allow for more experimentation and parameter tuning.
