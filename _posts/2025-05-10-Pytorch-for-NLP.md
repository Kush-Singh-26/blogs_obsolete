---
layout: post
title: "3. NLP: Leveraging Pytorch for NLP"
date: 2025-05-10
tags: [NLP]
---

## Loading Custom Data

- Data can be available in many formats like `csv`, `json` or `plain text`.
- Let's look at how to handle each type of data.

### CSV data

- Given data is like :

```c
text,label
"I love deep learning",1
"NLP is fun",0
"Transformers are powerful",1
"PyTorch makes it easy",0
```

It can be processed using `pandas` library.

```python
import pandas as pd
df = pd.read_csv('data.csv')
texts = df['text'].astype(str).tolist() # list of text strings
labels = df['label'].tolist() # list of labels (e.g. integers or categories)
```

This will give 2 lists `texts` and `labels`.

### JSON

- Given data is like :

```json
[
  {"text": "I love deep learning", "label": 1},
  {"text": "NLP is fun", "label": 0},
  {"text": "Transformers are powerful", "label": 1},
  {"text": "PyTorch makes it easy", "label": 0}
]
```

It can be processed using the `json` module

```python
import json
with open('data.json') as f:
    records = json.load(f) # records = list of dict.
texts = [rec['text'] for rec in records]
labels = [rec['label'] for rec in records]
```

Alternatively, `df = pd.read_json('data.json)` can also be used.


### Plain Text

- Assuming there are 2 different files for text and labels

```python
texts = []
with open('data.txt', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            texts.append(line)
```

- Same can be done for labels file too.

## Preprocessing text with NLTK

- Using `nltk` (Natural Language Toolkit) library for preprocessing textual data.
- It provides various methods to clean the raw text before converting to tenors.

### Tasks to be performed :

| | **Task** | **Purpose** |
| ---- | ---- | ----- | 
| 1 | *Lower casing* | for uniformity |
| 2 | *Tokenization* | Split text into tokens(words) |
| 3 | *Punctuation* Removal | Doesn't add any meaningful information |
| 4 | Removing *stop words* | Remove common, less informative words (eg. the, is, a)  |
| 5 | *Stemming* | Reduce word to their root form (may not be acurate words) |
| 6 | *Lemmatization* | Reduce words to their base/dictionary form |
