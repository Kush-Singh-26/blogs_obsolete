---
layout: post
title: "NLP Hugging Face"
date: 2025-03-07
tags: [data-science, machine-learning, kaggle, tutorials]
---
# 🤗 Tranformers

## `pipeline()`
- Easy way to use models for inference.
- it is the most basic element of the transformer library which connects a model directly to pre-processing and post-processing steps.
- Therefore, directly input the text and get the output.

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I am not feeling sad.")
```
**Output :**
> {'label': 'POSITIVE', 'score': 0.9993570446968079}

**Steps involved in working of the pipeline :**    
1) Input is preprocessed into a format the model can understand.    
2) Preprocessed inputs are passed to the model.     
3) Output of the model is post-processed.    

### Zero-shot classification
- classify texts that have not been labelled.
- zero-shot because there is no need to fine tune the model to use
- It allows to specify which labels to use for the classification, so don’t have to rely on the labels of the pretrained model.

``` python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a swimming pool",
    candidate_labels=["water", "politics", "running"],
)
```
 **Output :**
> {'sequence': 'This is a swimming pool',
> 'labels': ['water', 'running', 'politics'],   
> 'scores': [0.9990941882133484, 0.0006223419331945479, 0.0002834165934473276]}

### Text Generation
- provide a prompt and the model will auto-complete it by generating the remaining text.

``` python
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course about artificial intelligence, we will teach you how to")
```
**Output :**
>[{'generated_text': 'In this course about artificial intelligence, we will teach you how to build a program to understand a wide variety of topics. For those of you familiar with machine learning techniques and deep learning, this course will show you how to develop an intelligent program to understand'}]

### Using a specified model
- choose a particular model from the Hub to use in a pipeline for a specific task.
``` python
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
```

**Output :**
> [{'generated_text': 'In this course, we will teach you how to use your knowledge over and over again, to help you in your life.'},     
> {'generated_text': 'In this course, we will teach you how to read both the English and the English; we will give you the basics and the fundamentals that make your'}]

## Mask-filling

```python
from transformers import pipeline

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)
```
**Output :**
>[{'sequence': 'This course will teach you all about mathematical models.',
>  'score': 0.19619831442832947,
>  'token': 30412,
>  'token_str': ' mathematical'},      
> {'sequence': 'This course will teach you all about computational models.',
>  'score': 0.04052725434303284,
>  'token': 38163,
>  'token_str': ' computational'}]

- The top_k argument controls how many possibilities are to be displayed.
- Here the model fills in the special `<mask>` word, which is often referred to as a _mask token_. 
- Other mask-filling models might have different mask tokens.

### Named Entity Recognition

```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```
**Output :**
>[{'entity_group': 'PER',
>  'score': 0.9981694,
>  'word': 'Sylvain',
>  'start': 11,
>  'end': 18},   
> {'entity_group': 'ORG',
>  'score': 0.9796019,
>  'word': 'Hugging Face',
>  'start': 33,
>  'end': 45},    
> {'entity_group': 'LOC',
>  'score': 0.9932106,
>  'word': 'Brooklyn',
>  'start': 49,
>  'end': 57}]

- model finds which parts of the input text correspond to entities such as persons, locations, or organizations.

### Question-answering

```python
from transformers import pipeline

question_answerer = pipeline("question-answering")
question_answerer(
    question="Who is NASA located?",
    context="My name is Kush and I work at NASA in Brooklyn",
)
```
**Output :**
> {'score': 0.8570486903190613, 'start': 38, 'end': 46, 'answer': 'Brooklyn'}

### Summarization

```python
from transformers import pipeline

summarizer = pipeline("summarization")
summarizer(
    """
    Natural Language Processing (NLP) is a field of artificial intelligence (AI) that enables computers to understand, interpret, and generate human language. It combines linguistics and machine learning techniques to process text and speech data efficiently. NLP powers applications such as chatbots, machine translation, sentiment analysis, and voice assistants like Siri and Alexa.  

Key techniques in NLP include tokenization, stemming, lemmatization, named entity recognition (NER), and part-of-speech (POS) tagging. More advanced models, such as transformer-based architectures like BERT and GPT, have revolutionized NLP by enabling contextual understanding and generating human-like text.  

Despite its advancements, NLP faces challenges such as handling ambiguity, understanding context, and ensuring fairness in language models. Researchers continuously work on improving accuracy and reducing biases in AI-driven language systems. As NLP evolves, it is expected to enhance human-computer interaction, making technology more accessible and intuitive across various domains, including healthcare, education, and customer service.
    """
)
```

**Output :**
> [{'summary_text': ' Natural Language Processing (NLP) is a field of artificial intelligence that enables computers to understand, interpret, and generate human language . It combines linguistics and machine learning techniques to process text and speech data efficiently . NLP powers applications such as chatbots, machine translation, sentiment analysis and voice assistants like Siri and Alexa .'}]

### Translation

```python
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-hi-en")
print(translator("मेरा नाम कुश है। मैं नेचुरल लैंग्वेज प्रोसेसिंग की पढ़ाई कर रहा हूँ।"))
```

**Output :**
>[{'translation_text': "My name's Kusash. I'm studying Nechal Language Processing."}]