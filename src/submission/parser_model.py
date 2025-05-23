#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .parser_utils import load_and_preprocess_data


class ParserModel(nn.Module):
    """ Feedforward neural network with an embedding layer and single hidden layer.
    The ParserModel will predict which transition should be applied to a
    given partial parse configuration.

    PyTorch Notes:
        - Note that "ParserModel" is a subclass of the "nn.Module" class. In PyTorch all neural networks
            are a subclass of this "nn.Module".
        - The "__init__" method is where you define all the layers and their respective parameters
            (embedding layers, linear layers, dropout layers, etc.).
        - "__init__" gets automatically called when you create a new instance of your class, e.g.
            when you write "m = ParserModel()".
        - Other methods of ParserModel can access variables that have "self." prefix. Thus,
            you should add the "self." prefix layers, values, etc. that you want to utilize
            in other ParserModel methods.
        - For further documentation on "nn.Module" please see https://pytorch.org/docs/stable/nn.html.
    """

    def __init__(self, embeddings, n_features=36,
                 hidden_size=200, n_classes=3, dropout_prob=0.5):
        """ Initialize the parser model.

        @param embeddings (Tensor): word embeddings (num_words, embedding_size)
        @param n_features (int): number of input features
        @param hidden_size (int): number of hidden units
        @param n_classes (int): number of output classes
        @param dropout_prob (float): dropout probability
        """
        super(ParserModel, self).__init__()
        torch.manual_seed(0)
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.pretrained_embeddings = nn.Embedding(embeddings.shape[0], self.embed_size)
        self.pretrained_embeddings.weight = nn.Parameter(torch.tensor(embeddings))

        ### TODO:
        ###     1) Construct `self.embed_to_hidden` linear layer
        ###     2) Construct `self.dropout` layer.
        ###     3) Construct `self.hidden_to_logits` linear layer
        ###
        ### Please see the following docs for support:
        ###     Linear Layer: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        ###     Dropout: https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#torch.nn.Dropout
        ### START CODE HERE (~3 Lines)
        self.embed_to_hidden = nn.Linear(self.n_features * self.embed_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.hidden_to_logits = nn.Linear(self.hidden_size, self.n_classes)
        ### END CODE HERE

        self.reset_parameters()

    def reset_parameters(self) -> None:

        ### TODO:
        ###     1) Initialize `self.embed_to_hidden` linear layer's weight matrix
        ###         with the `nn.init.xavier_uniform_` function with `gain = 1` (default)
        ###     2) Initialize `self.hidden_to_logits` linear layer's weight matrix
        ###         with the `nn.init.xavier_uniform_` function with `gain = 1` (default)
        ###
        ### Note: Here, we use Xavier Uniform Initialization for our Weight initialization.
        ###         It has been shown empirically, that this provides better initial weights
        ###         for training networks than random uniform initialization.
        ###         For more details checkout this great blogpost:
        ###             http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization 
        ### Hints:
        ###     - You can access the weight matrix via:
        ###         linear_layer.weight
        ###     - Make sure you initialize the layers in the mentioned order above to comply with the autograder
        ###
        ### Please see the following docs for support:
        ###     Linear Layer: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        ###     Xavier Init: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
        pass
        ### START CODE HERE (~2 Lines)
        nn.init.xavier_uniform_(self.embed_to_hidden.weight, gain=1)
        nn.init.xavier_uniform_(self.hidden_to_logits.weight, gain=1)
        ### END CODE HERE

    def embedding_lookup(self, t):
        """ Utilize `self.pretrained_embeddings` to map input `t` from input tokens (integers)
            to embedding vectors.

            PyTorch Notes:
                - `self.pretrained_embeddings` is a torch.nn.Embedding object that we defined in __init__
                - Here `t` is a tensor where each row represents a list of features. Each feature is represented by an integer (input token).
                - In PyTorch the Embedding object, e.g. `self.pretrained_embeddings`, allows you to
                    go from an index to embedding. Please see the documentation (https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding)
                    to learn how to use `self.pretrained_embeddings` to extract the embeddings for your tensor `t`.

            @param t (Tensor): input tensor of tokens (batch_size, n_features)

            @return x (Tensor): tensor of embeddings for words represented in t
                                (batch_size, n_features * embed_size)
        """
        ### TODO:
        ###     1) Use `self.pretrained_embeddings` to lookup the embeddings for the input tokens in `t`.
        ###     2) After you apply the embedding lookup, you will have a tensor shape (batch_size, n_features, embedding_size).
        ###         Use the tensor `view` method to reshape the embeddings tensor to (batch_size, n_features * embedding_size)
        ###
        ### Note: In order to get batch_size, you may need use the tensor .size() function:
        ###         https://pytorch.org/docs/stable/tensors.html#torch.Tensor.size
        ###
        ###  Please see the following docs for support:
        ###     Embedding Layer: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html 
        ###     View: https://pytorch.org/docs/stable/tensor_view.html
        ### START CODE HERE (~1-3 Lines)
        x = self.pretrained_embeddings(t)
        x = x.view(x.size(0), -1)
        ### END CODE HERE
        return x

    def forward(self, t):
        """ Run the model forward.

            Note that we will not apply the softmax function here because it is included in the loss function nn.CrossEntropyLoss

            PyTorch Notes:
                - Every nn.Module object (PyTorch model) has a `forward` function.
                - When you apply your nn.Module to an input tensor `t` this function is applied to the tensor.
                    For example, if you created an instance of your ParserModel and applied it to some `t` as follows,
                    the `forward` function would called on `t` and the result would be stored in the `output` variable:
                        model = ParserModel()
                        output = model(t) # this calls the forward function
                - For more details checkout: https://pytorch.org/docs/stable/nn.html#torch.nn.Module.forward 

        @param t (Tensor): input tensor of tokens (batch_size, n_features)

        @return logits (Tensor): tensor of predictions (output after applying the layers of the network)
                                 without applying softmax (batch_size, n_classes)
        """
        ### TODO:
        ###     1) Apply `self.embedding_lookup` to `t` to get the embeddings
        ###     2) Apply `embed_to_hidden` linear layer to the embeddings
        ###     3) Apply relu non-linearity to the output of step 2 to get the hidden units.
        ###     4) Apply dropout layer to the output of step 3.
        ###     5) Apply `hidden_to_logits` layer to the output of step 4 to get the logits.
        ###
        ### Note: We do not apply the softmax to the logits here, because
        ### the loss function (torch.nn.CrossEntropyLoss) applies it more efficiently.
        ###
        ### Please see the following docs for support:
        ###     ReLU: https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html#torch.nn.functional.relu 
        ###  START CODE HERE (~3-5 lines)
        x = self.embedding_lookup(t)
        hidden = F.relu(self.embed_to_hidden(x))
        hidden = self.dropout(hidden)
        logits = self.hidden_to_logits(hidden)
        ### END CODE HERE
        return logits
