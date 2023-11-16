# Improving Zero-shot Visual Question Answering via Large Language Models with Reasoning Question Prompts

## Repository structure
We have the important files with the following structure. We omit some datasets and dependencies due to the loading size limit.

```
+-- code
    +-- rq_prompting_step1.py # The code for answer generation
    +-- rq_prompting_step2.py # The code for answer selection
    +-- src # The code for question edit
+-- data
    +-- okvqa # Processed OKVQA data
    +-- aokvqa # Processed OKVQA data
    +-- caption # Captions for VQA datasets
    +-- cache # Cache to save queried prompts
    +-- edit_results # The folder to save edit questions
+-- results # The folder to save output results
```


## Setup
This implementation is based on python3. 

The question edition code is largely modified based on [Edit-Unsup-TS](https://github.com/ddhruvkr/Edit-Unsup-TS), you need to have a [CoreNLP Server running on port 9000](https://stanfordnlp.github.io/CoreNLP/download.html) in code/src/. We ultized well-trained model on Wikilarge to conduct inference on the VQA datasets, the trained word2vec model can be found [here](https://drive.google.com/drive/folders/17dbLIZpCj3taAD1xbea9OQh4hvPee6Oc), should be put in code/src/Wikilarge folder. The glove word embedding can be found [here](https://nlp.stanford.edu/projects/glove/), should be put in code/src/Embeddings folder.

The question answering part should be filled with valid apikey in the argument.

## Run pipeline

1. Enter in code/src and modify the config to specify the dataset and importance of different factors.

```
cd code/src \
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
-preload tokenize,ssplit,pos,lemma,parse,depparse \
-status_port 9000 -port 9000 -timeout 15000\
python3 src/main.py
```

2. Enter in code/, Specify the config for question answering and run question answering based on the edit questions

```
cd code/ \
python3 rq_prompting_step1.py \
python3 rq_prompting_step2.py 
```


## Citation

if you find this useful for your work, please cite:

>@inproceedings{lan:mm2023,\
>    author = {Lan, Yunshi and Li, Xiang and Liu, Xin and Li, Yang and Qin, Wei and Qian, Weining},\
>    title = {Improving Zero-Shot Visual Question Answering via Large Language Models with Reasoning Question Prompts},\
>    year = {2023},\
>    isbn = {9798400701085},\
>    publisher = {Association for Computing Machinery},\
>    address = {New York, NY, USA},\
>    url = {https://doi.org/10.1145/3581783.3612389}, \
>    doi = {10.1145/3581783.3612389},\
>    pages = {4389–4400},\
>    numpages = {12},\
>    keywords = {zero-shot evaluation, large language models, visual question answering},\
>    location = {Ottawa ON, Canada},\
>    series = {MM '23}\
>}