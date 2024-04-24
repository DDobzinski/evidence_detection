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

This is a classification model that was trained to
      detect whether a claim is supported by a piece of evidence.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model employs a custom Transformer encoder architecture, similar to BERT, 
    fine-tuned on a dataset of 23k claim-evidence pairs. The model processes text inputs to classify 
    whether the evidence supports the claim, with an additional classification head for output prediction.

- **Developed by:** Zhe Khang Chong and Daniel Dobzinski
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Transformers
- **Finetuned from model [optional]:** d2l Transformer Encoder

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://github.com/d2l-ai/d2l-en
- **Paper or documentation:** https://d2l.ai/

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

The model uses 23K claim-evidence pairs as training data. These pairs are concatenated into a single sequence, allowing the model to consider the relationship between the claim and the evidence as a unified context. Following this, NLTK’s word_tokenize function splits the text into words based on spaces and punctuation. Each word token is then replaced with a corresponding vector from a pre-trained GloVe embedding. If a word is not found in the GloVe dictionary, it is substituted with a zero vector of size 100. This ensures that all input sequences are of uniform length.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 0.05
      - delta: 1e-03
      - gamma: 0.25
      - train_batch_size: 16
      - eval_batch_size: 16
      - predict_batch_size: 32
      - num_hiddens: 100
      - ffn_num_hiddens: 256
      - num_heads: 4
      - num_blks: 2
      - dropout: 0.1
      - num_classes: 2
      - num_epochs: 7

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 19 minutes
      - duration per training epoch: 1.3 minutes
      - model size: 730KB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

A subset of the development set provided, amounting to 6K pairs. Claim and evidence for some pairs are shuffled around to ensure the model generalizes well.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision
      - Recall
      - F1-score
      - Accuracy

### Results

The best model obtained an accuracy of 81%, precision of 77.5%, recall of 74%, F1-score of 75%, training loss of 41% and validation loss of 42%.

## Technical Specifications

### Hardware


      - RAM: at least 8 GB
      - Storage: at least 2GB

### Software


      - Torchvision 0.17.1+cu121
      - Pytorch 2.2.1+cu121
      - nltk 3.8.1
      - numpy 1.25.2
      - pandas 2.0.3
      - d2l 1.0.3

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any inputs (concatenation of two sequences) longer than
      100 tokens will be truncated by the model, as positional encoding will only generate 100 possible positional values. This model was trained
      on 23/04/2024, thus fine tuning may be needed in the future if language changed significantly. The model is trained on 23k claim evidence pairs
      which could lead to biases from this dataset carrying over to the model predictions.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

Hyperparameters were optimized based on a grid search over a subset of the training data. Model saving was 
    implemented at each epoch to monitor training progress and allow for recovery in case of interruption.
