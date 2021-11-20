# Machine Learning Chatbot

This is a subject project in my university.

## Usages:

### Install Miniconda

```
   86  wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh
   87  bash Miniconda3-py39_4.10.3-Linux-x86_64.sh
   90  miniconda3/bin/conda init
   93  export PATH="/root/miniconda/bin:$PATH"
   94  conda
```
### Clone this project

```
   95  git clone https://github.com/nhamtphat/ml-chatbot
   96  cd ml-chatbot/
```
### Create conda env
Without env.yml:
``` 
   97  conda create
   98  conda create --help
   99  conda create -n tf tensorflow
  100  conda activate tf
  101  python app.py
  102  conda install nltk
  103  python app.py
  104  conda install tflearn
  105  pip install tflearn
  106  python app.py
  107  pip install flask
  108  conda install flask
  109  python app.py
```
With env.yml
```
conda env create -f environment.yml
conda activate tf
python app.py
```
