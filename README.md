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
### D5
#### Automatic prep + validation
1. Switch to the `data_analysis` module:
```cmd
cd data_analysis
```
2. Run the controller validation script which will run everything for all datasets and partitions:
```cmd
python validate.py
```
#### Manual prep + validation
1. Switch to the `data_analysis` module:
```cmd
cd data_analysis
```
2. For each dataset and partition, prepare the columns for data analysis
```cmd
python prepare_clean.py CNN-DailyMail Control
python prepare_clean.py CNN-DailyMail Treatment
python prepare_clean.py Samsum Control
python prepare_clean.py Samsum Treatment
python prepare_clean.py Xsum Control
python prepare_clean.py Xsum Treatment
```

3. Validate that analysis prep went through correctly for both scores and classifier results:
```cmd
python validate.py CNN-DailyMail Control
python validate.py CNN-DailyMail Treatment
python validate.py Samsum Control
python validate.py Samsum Treatment
python validate.py Xsum Control
python validate.py Xsum Treatment
```