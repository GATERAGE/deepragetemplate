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
```bash
python3 -m venv rage
```
```bash
source rage/bin/activate
```
```bash
pip install -r requirements.txt
```
```bash
python rage.py --query "What are the main advantages of DeepSeek R1?" --output-format markdown
```
