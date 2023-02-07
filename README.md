# ADDMU: Detection of Far-Boundary Adversarial Examples with Data and Model Uncertainty Estimation
Code for our EMNLP 2022 paper. The framework and metrics are adapted from https://github.com/bangawayoo/adversarial-examples-in-text-classification

## Dependencies
Python >= 3.6

Pytorch=1.8.1

Install python requirments via requirments file: <code> pip install -r requirements.txt </code>

## Data
We use [TextAttack](https://github.com/QData/TextAttack) to generate attack data of the four attacks,  TextFooler, BAE, Pruthi, and TextBugger. If you want to generate your own adversarial data, please refer to their repos. We also provide [here](https://drive.google.com/drive/folders/1-8EA1HjlCs6xDdRlRrLFp5UU9yGFALnr?usp=share_link) with some of the data we generate, including both regular and far-boundary data. Please download the whole folder and put them under the main directory.

## Usage
The experiments can be reproduced by simply running the following shell script:

<code> bash run_test_sst2.sh </code>

This is the example script for sst2. Changing the datasets, attack types, and detectors with the following options.

Options for the datasets are sst2, imdb, ag-news, and snli, which are listed with the <code> DATASET </code> variable.

Options for the type of attacks are textfooler, bae, pruthi, and textbugger, which are listed with the <code> RECIPE </code> variable. Also, use  <code> *_high_confidence_0.9 </code>, such as  <code> textfooler_high_confidence_0.9 </code> for far boundary version of attacks.

Options for the detectors are our proposed method <code>ue</code>, and two other baselines, <code>ppl</code> and <code>rde</code>
