# README

## Overview

This README provides an overview of the workflow for processing RAG datasets, including the following scripts: `preprocess.py`, `split.py`, `filter.py`, `embed.py`, `retrieve.py`, `evaluate.py`, and `get_results.py`. Each script plays a specific role in preparing and evaluating the dataset.



## Request

### todo







## Preprocessing

### Description

The `preprocess.py` script is designed to preprocess RAG datasets by tokenizing text, removing stopwords, and saving the cleaned data into a new file format.

### Usage

You can download the **punkt** tokenizer and **stopwords** using the following code:

```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
```

After downloading, you can run the preprocessing script from the command line with the following command:

```BASH
python RAG/preprocess.py --dataset_path <DATASET_PATH> --output_path <OUTPUT_PATH> --files <FILE_NAMES>
```

### Example Command

```BASH
python RAG/preprocess.py --dataset_path data/ --output_path data/ --files COMA
```

### File Naming

The output files will be named following the convention `<original_file_name>_preprocessed.jsonl`.



## Splitting

### Description

The `split.py` script is designed to split the preprocessed data into smaller chunks, ensuring that no chunk exceeds a specified maximum token length.

### Usage

You can run the splitting script from the command line:

```
python RAG/split.py --input_path <INPUT_PATH> --output_path <OUTPUT_PATH>
```

### Example Command

```
python RAG/split.py --input_path data/ --output_path data/
```

### File Naming

The output files will be named following the convention `<original_file_name>_split.jsonl`.



## Filtering

### Description

The `filter.py` script filters the split data based on the overlap between the RAG chunks and the question. Only chunks that share common words with the question are retained.

### Usage

You can run the filtering script from the command line:

```
python RAG/filter.py --input_path <INPUT_PATH> --output_path <OUTPUT_PATH>
```

### Example Command

```
python RAG/filter.py --input_path data/ --output_path data/
```

### File Naming

The output files will be named following the convention `<original_file_name>_filter.jsonl`.



## Embedding

### Description

The `embed.py` script is responsible for generating embedding vectors for the filtered data using a specified embedding model.

### Usage

You can run the embedding script from the command line:

```
python RAG/embed.py --path <PATH>
```

### Example Command

```
python RAG/embed.py --path data/
```

### File Naming

The output files will be named following the convention `<original_file_name>_embedding.jsonl`.



## Retrieval

### Description

The `retrieve.py` script retrieves relevant documents based on a query, using embeddings and reranking techniques.

### Usage

You can run the retrieval script from the command line:

```
python RAG/retrieve.py --path <PATH> --task <TASK_NAME>
```

### Example Command

```
python RAG/retrieve.py --folder data/ --task COST
```

### File Naming

The output file will be named following the convention `<task>_rag.jsonl`.



## Evaluate RAG Quality(Optional)

### Usage

You can run the relevance score analysis script from the command line to calculate the average relevance score and the proportions of scores that are above 0.8 and below 0.2.

```bash
python RAG/relevance_score.py --path <DATA_FOLDER>
```

### Example Command

```bash
python RAG/relevance_score.py --path data/
```



## Evaluate Model

### Description

The `evaluate.py` script evaluates the performance of the retrieval results based on the gold answers.

### Usage

You can run the evaluation script from the command line:

```bash
python RAG/evaluate.py --task <TASK> -prompt <PROMPT_TYPE> --model <MODEL> --year <YEAR> --path <PATH>
```

### Example Command

```bash
python RAG/evaluate.py --task ALL -prompt BASE --model gpt-3.5-turbo --year 2023 --path data/
```



## Get Results

### Description

The `get_results.py` script aggregates and formats the evaluation results.

### Usage

You can run the get results script from the command line:

```bash
python RAG/get_results.py --year <YEAR> --path <PATH>
```

### Example Command

```bash
python RAG/get_results.py --year 2023 --path data/
```





## Complete Workflow Example

After the **request**, here’s a complete workflow example that shows how to go from preprocessing to getting results:

Reminder: The input path should match the output path of the last step.

```bash
cd NewTerm++/
#Step 1: Preprocessing
python RAG/preprocess.py --dataset_path data/ --output_path data/

# Step 2: Splitting
python RAG/split.py --input_path data/ --output_path data/

# Step 3: Filtering
python RAG/filter.py --input_path data/ --output_path data/

# Step 4: Embedding
python RAG/embed.py --path data/

# Step 5: Retrieval
python RAG/retrieve.py --path data/ --task COMA COST CSJ

# Optional Step 5.1: Evaluate RAG Quality
python RAG/relevance_score.py --path data/

# Step 6: Evaluation
python RAG/evaluate.py --task ALL -prompt BASE --model gpt-3.5-turbo --year 2023 --path data/

# Step 7: Get Results
python RAG/get_results.py --year 2023 --path data/
```



## Using Environment Variables

For ease of use, you may want to set an environment variable for your data folder:

```bash
export REQUEST_FOLDER=/path/to/_clean_search/
export DATA_FOLDER=/path/to/outputdata/
```



Then, you can use the environment variable to run the scripts:

```bash
cd NewTerm++/
#Step 1: Preprocessing
python RAG/preprocess.py --dataset_path $REQUEST_FOLDER --output_path $DATA_FOLDER

# Step 2: Splitting
python RAG/split.py --input_path $DATA_FOLDER --output_path $DATA_FOLDER

# Step 3: Filtering
python RAG/filter.py --input_path $DATA_FOLDER --output_path $DATA_FOLDER

# Step 4: Embedding
python RAG/embed.py --path $DATA_FOLDER

# Step 5: Retrieval
python RAG/retrieve.py --path $DATA_FOLDER --task COMA COST CSJ

# Optional Step 5.1: Evaluate RAG Quality
python RAG/relevance_score.py --path $DATA_FOLDER

# Step 6: Evaluation
python RAG/evaluate.py --task ALL -prompt BASE --model gpt-3.5-turbo --year 2023 --path $DATA_FOLDER

# Step 7: Get Results
python RAG/get_results.py --year 2023 --path $DATA_FOLDER
```



