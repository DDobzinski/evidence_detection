---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/DDobzinski/evidence_detection

---

# Model Card for b03791zc-h59035dd-ED

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model trained to detect whether the evidence is relevant to the claim. The model employs a deep learning-based approach that does not use transformer architectures for evidence detection. Instead, it utilizes a Hierarchical Attention Network, which includes a GRU Layer, Coherence Attention, Entailment Attention, and a Linear Classifier.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is based on a Hierarchical Attention Network developed in the paper linked below. After text processing, the embeddings are fed into a Gated Recurrent Unit (GRU), a type of recurrent neural network. The GRU processes the sequence of embeddings to produce a new sequence that captures dependencies across the text. There are two attention mechanisms: Coherence Attention and Entailment Attention. Coherence Attention assesses the coherence of the information within the claim and the evidence. It assigns weights to different parts of the text, highlighting areas that are more relevant or coherent within the context of the entire text. This step helps the model focus on the most important parts of the claim and evidence. Entailment Attention focuses on determining how well the evidence supports or contradicts the claim. This involves comparing embeddings and their sequences to understand the logical and semantic relationships between the claim and the evidence. The outputs from the coherence and entailment attention mechanisms are combined to form a comprehensive context vector for each set of claim and evidence. The combined context vector is then passed through a classifier.

- **Developed by:** Daniel Dobzinski and Zhe Khang Chong
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Attention-based Sequence Processing Model 
- **Finetuned from model [optional]:** [More Information Needed]

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** http://nlp.stanford.edu/data/glove.6B.zip
- **Paper or documentation:** https://ink.library.smu.edu.sg/sis_research/4557/

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

The model uses 23K claim-evidence pairs as training data. These pairs are concatenated into a single sequence, allowing the model to consider the relationship between the claim and the evidence as a unified context. Following this, NLTK’s word_tokenize function splits the text into words based on spaces and punctuation. Each word token is then replaced with a corresponding vector from a pre-trained GloVe embedding. If a word is not found in the GloVe dictionary, it is substituted with a zero vector of size 100. This ensures that all input sequences are of uniform length.
    

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 1e-04
      - embedding_dim: 100
      - train_batch_size: 16
      - eval_batch_size: 16
      - predict_batch_size: 32
      - GRU_hidden_size: 256
      - max-length: 256
      - num_epochs: 10

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 2 hours
      - duration per training epoch: 20 minutes
      - model size: 2MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

For evaluating model performance, both training and validation datasets are used. The function iterates over the entire dataset multiple times, as defined by num_epochs. After each training epoch, the model is evaluated.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Training Loss
      - Validation Accuracy
      - Validation Loss

### Results

The model achieved a validation accuracy of 83% and a validation loss of 36%. Additionally, it recorded a training loss of 34%.

## Technical Specifications

### Hardware


      - RAM: at least 8 GB
      - Storage: at least 2GB,

### Software


      - Torchvision 0.17.1+cu121
      - Pytorch 2.2.1+cu121
      - nltk 3.8.1
      - numpy 1.25.2
      - pandas 2.0.3

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The model's performance heavily depends on the word embeddings used, such as GloVe. If these embeddings are trained on datasets that do not represent the diversity of the language, the model might inherit biases. Additionally, the interpretability provided by attention weights can be misleading. The model’s "focus," as derived from attention weights, does not always correspond to human notions of relevance or importance. Moreover, the model operates on fixed-length input sequences.
    

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The approach does not follow paper exactly because it is aimed at evidence comprising of multiple sentences. Experiments were conducted to explore the use of BERT embeddings; however, the training process proved to be incredibly long. To accelerate development, a switch to GloVe embeddings was made.
