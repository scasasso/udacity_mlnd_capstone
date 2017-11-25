## Structure
The repository is organized in files/directories:
- `README.md`: it's this file
- `proposal.pdf`: it's the file with the proposal for the Capstone project
- `fake_news_detector.ipynb`: it's the Jupyter Notebook with the analysis and results
- `data`: it contains the zip file with the datasets
- `scripts`: it contains the necessary scripts to reproduce the results
- `proposal`: it contains the files to write `proposal.pdf`

## Instructions
### Build the dataset
The datasets are already available under the `data` directory (`data.zip`).  
However, if you would like to reproduce it from scratch, follow the instructions below:  
- download and unzip the dataset for fake news from [Kaggle](https://www.kaggle.com/mrisdal/fake-news);     
- download the dataset for real news from [webhose.io](https://webhose.io/datasets), 
the one under "English news articles"; unzip it in a separate directory (about 500 thousands file will be extracted)  
- run the script `scripts/crawl_real_news.py` to transform the json files to one single csv; run:  
```bash
python scripts/crawl_real_news.py --help
```
to inspect the command line options.  
At this point, you should have two files: `fake.csv` and `real.csv`: put them under the `data` directory, 
as the scripts will read them from there.

### Exploratory plots
In order to reproduce the exploratory plots included in the final report, one has to run the following command:  
```bash
cd scripts
python explore.py
``` 
The script produces 9 plots in pdf format. 

### Fit models
The 3 classifiers defined in the final report are fitted, optimised and validated 
running the script `scripts/fit_separate.py`:   
```bash
cd scripts
python fit_separate.py
```
The script produces in output 3 pickle files, each one containing a different model. 
N.B.: the script produces several error messages related to failed fit: this is due to 
erroneous combinations of parameters. However, these errors are ignored and only the good 
combinations of parameters are retained.   
N.B.2: the script takes a long time to run (about 1 hour on a 4 core machine), due to the parameter optimisation step.

### Fit ensembles
__N.B.: In order to run this step one has to run the separate models before (see Section "Fit models")
and have the files `mnb_opt.pkl`, `lr_opt.pkl`, `rndf_opt.pkl` in the same directory__  
The ensemble classifiers are fitted with the script `scripts/fit_ensemble.py`:   
```bash
cd scripts
python fit_ensemble.py
``` 
The scripts produces in output 2 pickle files, containing the ensemble models. 

### Run sensitivity test 
__N.B.: In order to run this step one has to run the ensemble models before (see Section "Fit ensemble")
and have the file `en2_opt.pkl` in the same directory__  
The sensitivity test described in the final report is run with the script `scripts/test_sensitivity.py`:  
```bash
cd scripts
python test_sensitivity.py
```
The script writes the results of the sensitivity test in the standard output. 

### Plot score vs. number of samples
__N.B.: In order to run this step one has to run the separate models before (see Section "Fit models")
and have the files `lr_opt.pkl`, `rndf_opt.pkl` in the same directory__  
The plot described in the "Free-form visualization" section in the final report 
is produced with the script `scripts/plot_sample_dependency.py`:  
```bash
cd scripts
python plot_sample_dependency.py
```
The script outputs a pdf file with the plot of the scores of the ensemble classifier as a function 
of the number of samples in the training dataset. 