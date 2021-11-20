# Machine Learning Chatbot

This is a subject project in my university.

## Usages:

### 1. Install Miniconda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
bash Miniconda3-py39_4.10.3-Linux-x86_64.sh
miniconda3/bin/conda init
export PATH="/root/miniconda/bin:$PATH"
conda
```
### 2. Clone this project

```
git clone https://github.com/nhamtphat/ml-chatbot
cd ml-chatbot/
```
### 3. Create conda env
With env.yml
```
conda env create -f environment.yml
conda activate tf
python app.py
```
Without env.yml:
``` 
conda create -n tf tensorflow
conda activate tf
conda install nltk
pip install tflearn
conda install flask
python app.py
```
