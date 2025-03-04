```txt
deeprage/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── document.py
│   ├── parser.py
│   ├── store.py
│   ├── retriever.py
│   ├── generator.py
│   └── utils.py
├── tests/
│   └── __init__.py
├── knowledge/
│   ├── markdown/
│   ├── json/
│   ├── index/
│   ├── cache/
│   └── responses/
├── requirements.txt
└── rage.py
```

tested on Ubuntu 22.04LTS with Python 3.11

```bash
pip install --upgrade pip setuptools
```
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

```bash
pip --version
```
```bash
pip check
```


sudo apt update && sudo apt install python3.11 python3.11-venv

```bash
python3.11 -m venv rage
```
```bash
source rage/bin/activate
```
```bash
pip install --no-cache-dir -r requirements.txt
```
```bash
streamlit run rage.py```
