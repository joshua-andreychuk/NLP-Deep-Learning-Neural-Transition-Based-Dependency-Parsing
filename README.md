# NLP-Deep-Learning-Neural-Transition-Based-Dependency-Parsing

## Overview
This repository contains my implementation where I built a neural dependency parser using PyTorch. The system processes sentences with a transition‑based algorithm, extracting rich structural features and training a feedforward network to predict parser actions, achieving robust Unlabeled Attachment Score (UAS) on the Penn Treebank.

## Key Features & Highlights
- **Transition Mechanics Implementation** (`parser_transitions.py`): Completed `PartialParse` and `minibatch_parse`, enabling shift, left‑arc, and right‑arc operations and batch parsing of multiple sentences for efficiency.
- **Feature Extraction Pipeline** (`parser_utils.py`): Leveraged universal dependency features—stack top, buffer front, left/right children, POS tags, and dependency labels—to generate fixed‑size input vectors for the neural model.
- **Feedforward Parser Model** (`parser_model.py`): Defined a two‑layer neural network with pretrained embeddings, Xavier initialization, ReLU activations, dropout regularization, and a final classification layer to select the next transition.
- **Training Loop & Evaluation** (`train.py`): Implemented epoch‑level training with Adam optimization, cross‑entropy loss, and checkpointing upon best dev‑set UAS; reported average training loss and dev UAS after each epoch.
- **Modular Design & Autograder Compatibility** (`__init__.py`): Organized code into clean modules, adhering to autograder conventions for easy testing and submission.

## Results
| Metric         | Value     |
|---------------:|----------:|
| Dev UAS        |     88.68% |
| Train Loss     |      0.0666 |
| Test UAS       |     89.34% |

*These figures meet or exceed the assignment thresholds: dev UAS ≥87% and train loss ≤0.08.* ≥87% and train loss ≤0.08.*

## Project Structure
- **src/submission/parser_transitions.py**: Implements `PartialParse` and batch parsing logic.  
- **src/submission/parser_utils.py**: Loads data, builds vocabulary, extracts features, and manages minibatches.  
- **src/submission/parser_model.py**: Defines the neural network architecture for transition classification.  
- **src/submission/train.py**: Contains training and evaluation routines with logging of loss and UAS.  
- **src/submission/__init__.py**: Exposes core classes and functions for autograder integration.  

## Insights
- **Structured Prediction with Neural Networks**: Combining classic transition‑based parsing algorithms with neural feature learning yields both interpretability (via transitions) and adaptability (via embeddings).  
- **Importance of Feature Engineering**: Careful selection of dependency and POS features, along with pretrained word embeddings, significantly impacts parser accuracy.  
- **Efficiency via Minibatching**: Implementing `minibatch_parse` allows parallel parsing decisions, speeding up evaluation on large test sets without altering algorithmic correctness.

## Assignment Requirements Met
- Implemented all core components: transition functions (a), minibatch parsing (b), neural model initialization and forward pass (c), feature lookup (c), and training loop (c).  
- Trained the model to achieve the required dev UAS >87% and train loss <0.08, demonstrating effective convergence.  

