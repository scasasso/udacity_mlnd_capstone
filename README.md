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
```python
python scripts/crawl_real_news.py --help
```
to inspect the command line options.  
At this point, you should have two files: `fake.csv` and `real.csv`: put them under the `data` directory, 
as the notebook will read them from there.