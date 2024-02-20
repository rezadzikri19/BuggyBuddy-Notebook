# 🪲 BuggyBuddy Notebook

BuggyBuddy is an end-to-end data science project aimed at performing similarity search through a bug report database to assist users/operators in identifying duplicate bug reports. This facilitates faster problem resolution in software development.

## Data Sources

This project utilizes Firefox bug report data sourced via [Bugzilla API](). The dataset comprises structured information such as bug `id`, `duplicates`, `summary`, `description`, `status`, `resolution`, `platform`, `product`, `type`, `priority`, `severity`, `component`.

## Data Preprocessing

Before the raw report data is used for model training, data preprocessing is performed as depicted in the diagram below. Several relevant features in the Raw Report are aggregated into a text feature, which then undergoes cleaning and stop word removal processes.

The cleaned text data is then embedded using the pretrained model `all-MiniLM-L6-v2`, resulting in 384-dimensional vectors. From this data, text pairs are formed and labeled `"1"` if the text pair is the same/duplicate, while `"0"` if the text pair is unique.

```mermaid
flowchart LR;
    preprocessing["`**Preprocessing**:
    • drop unused features
    • remove duplicates
    • aggregate text features
    • clean sentences
    • remove stopwords
    • sentence embedding (all-MiniLM-L6-v2)
    • reports pairing & labeling`"]
    style preprocessing text-align:center

    report{{"`**Raw Reports**
    *- status
    - product
    - component
    - description
    - resolution
    - severity
    - etc..*`"}}
    style report text-align:center

    report_1{{"`**Preprocessed Reports**
    *- embedded_text_left (384 vectors)
    - embedded_text_right (384 vectors)
    - label (0 for unique, 1 for duplicated)*`"}}
    style report_1 text-align:center

    report --> preprocessing
    preprocessing --> report_1
```

## Models Used

The model used is a Siamese network trained to detect whether two texts are the same (duplicate) or not. The model takes two inputs (embedded text) which are then transformed using a *Dense (fully connected)* layer (256 nodes) to produce a vector. The similarity of the vectors is then calculated using *Normalized Dot Layer*, and finally, a *Single Dense Node* with sigmoid activation is used to provide the output of the predicted similarity score.

```mermaid
flowchart LR;
    report_l{{"`**embedded_text_left**
    384 vectors`"}}
    style report_l text-align:center

    report_r{{"`**embedded_text_right**
    384 vectors`"}}
    style report_r text-align:center

    input_layer["`**Dense_Layer**
    256 nodes (linear)
    ⬤
    ⬤
    ⬤
    .
    .
    .
    ⬤
    ⬤
    ⬤`"]
    style input_layer text-align:center

    similar["`**Normalized_Dot_Layer**
    cosine_similarity
    ⬤`"]
    style similar text-align:center

    output_layer["`**Dense_Layer**
    1 node (sigmoid)
    ⬤`"]
    style output_layer text-align:center

    result{{"`**similarity score**
    1 - 0`"}}
    style result text-align:center

    report_l --> input_layer
    report_r --> input_layer

    input_layer --> similar
    similar --> output_layer
    output_layer --> result
```

In the context of similarity search, *we only utilize the Dense Layer as the projection layer to generate vectors* that can be used for similarity search.

## Code Structure

The BuggyBuddy project is organized as follows:

- `data/`: Contains raw data (fetched from API) and cached data (created during preprocessing).
- `models/`: Includes trained models.
- `bug_classification.ipynb`: Jupyter notebook for data exploration, model training, and evaluation.
- `requirements.txt`: List of dependencies for easy installation.

## Findings and Results

**Duplicate Bug Detection:**

- Successfully created duplicated bug reports with the following metric scores:
  - Precision: `0.92835`
  - Recall: `0.90968`
  - F1 Score: `0.91892`

**Similarity Search:**

- Utilizing [faiss](https://github.com/facebookresearch/faiss) for testing the performance of similarity search, successfully provided recommendations for similar bug reports with a recall@10 of `0.3301`.
