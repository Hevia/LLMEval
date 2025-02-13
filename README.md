# LLMEval

## Setup Instructions

1. Download and install Ollama: https://ollama.com/
2. Make sure Ollama is running when you run these models
    - The code will handle downloading the models for you, test using something small like SMOLLM2!!
3. Make sure Ollama is running when you run the code
4. Setup your virtual environment according to instructions below


Python version: `3.11.9`, you can use [pyenv](https://github.com/pyenv/pyenv) to manage your local python installs

### Linux/macOS

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

### Windows

1. Create a virtual environment:
```cmd
python -m venv venv
```

2. Activate the virtual environment:
```cmd
.\venv\Scripts\activate
```

3. Install requirements:
```cmd
pip install -r requirements.txt
```
### D3
#### Downloading Data
For SAM-Sum:
```cmd
curl -L -O https://huggingface.co/datasets/Samsung/samsum/resolve/main/data/corpus.7z 
7z x corpus.7z
```

For Webis:
```cmd
for i in {0..9};
    do curl -L -O https://huggingface.co/datasets/webis/tldr-17/resolve/refs%2Fconvert%2Fparquet/default/partial-train/000$i.parquet; 
done
```
#### EDA
- `samsum/test.json` ([Source](https://huggingface.co/datasets/Samsung/samsum))
    - 819 instances
    - 3 fields: `id`, `summary`, `dialogue`
    - dialogue ranges from 3-30 utterances (newline-separated)
    
- `webis/data.json` ([Source](https://huggingface.co/datasets/webis/tldr-17))
    - 3,848,330 instances in full
    - We can consider using the `partial-train` branch data (9 parquets) -- see download instructions above
    - relevant fields: `id`, `content`, `summary`, `subreddit`

A single parquet of the Webis data is massive compared to SAM-Sum test split. [Another repo](https://github.com/anna-kay/Reddit-summarization/blob/main/notebooks/filtering/Webis-TLDR-17_filtering.ipynb) includes analysis on duplicate rows and removing problematic rows, which we may want to do as well. We may also want to remove noisy/graphic data from certain subreddits, etc. (maybe only keep `TrueReddit`, or `AskReddit`?)