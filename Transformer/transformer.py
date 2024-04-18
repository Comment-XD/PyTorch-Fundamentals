import torch
import torch.nn as nn
import math
import numpy as np

class Positional_Encoding(nn.Module):
    def __init__(self, drop_value:float=0.1, d_model:int=4, max_length:int=5000):
        super().__init__()
        """
        drop_value (float) drop out value 
        n: (int) constant value 
        d_model (int): the dimension of the word embeddings 
        max_length (int): the total words in a sentence 
        
        """
        self.divisor = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000) / d_model)) # Creates the denominator component 
        self.drop_out = nn.Dropout(drop_value)
    
        self.k = torch.arange(0, max_length).unsqueeze(dim=1) # Creates the position elements 
        self.pe = torch.zeros(max_length, d_model) # Creates the empty array that will store the positional encodings
        
        self.pe[:, 0::2] = torch.sin(self.k * self.divisor) # For every row and even column, implement sin function
        self.pe[:, 1::2] = torch.cos(self.k * self.divisor) # For every row and odd column, implement cos function
        
        self.pe.squeeze(0)

    def forward(self, x):
        return self.drop_out(x + self.pe[:x.size(1)].requires_grad_(False))
        
class Scalar_Dot_Product_Attention(nn.Module):
    def __init__(self, d_model:int, dropout_value:int=0.1, mask=None) -> None:
        super().__init__()

        """
        d_model (int): the dimension of the word embeddings
        """
        
        self.d_model = d_model

        # Query, Key, Value Weights
        self.Qw = nn.Linear(d_model, d_model, bias=False)
        self.Kw = nn.Linear(d_model, d_model, bias=False)
        self.Vw = nn.Linear(d_model, d_model, bias=False)

	self.dropout = nn.Dropout(dropout_value)

    def forward(self, X:torch.Tensor()) -> torch.Tensor():
        """
        X (torch.Tensor): a Tensor that contains the sum between the Word Embeddings and Positional Encodings

        returns (torch.Tensor): Returns the attention score of the input X
        """
        Q = X @ self.Qw
        K = X @ self.Kw
        V = X @ self.Vw

        scaled_dot_prod = (Q @ K.permute(0,2,1)) / math.sqrt(d_model)
        attention_prob = scaled_dot_prod.softmax(dim=-1)

        attention_scores = self.dropout(attention_prob @ V)

        return attention_scores


class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model:int, heads:int=4, dropout_value:int=0.1, mask=None) -> None:
        super().__init__()

        """
        d_model (int): the dimension of the word embeddings
        heads (int): the number of heads 
        """

        # Dimension of Embedding
        self.d_model = d_model

        # Number of Heads in our Multi-Head-Attention
        self.heads = heads

        # Query, Key, Value Weights
        self.Qw = nn.Linear(d_model, d_model, bias=False)
        self.Kw = nn.Linear(d_model, d_model, bias=False)
        self.Vw = nn.Linear(d_model, d_model, bias=False)
        self.Ow = nn.Linear(d_model, d_model, bias=False)

        # 
        self.drop_out = nn.Dropout(p=dropout_value)

    def forward(self, X:torch.Tensor()) -> torch.Tensor():
        """
        X (torch.Tensor): a Tensor that contains the sum between the Word Embeddings and Positional Encodings

        Returns (torch.Tensor): Returns the attention score of the input X
        """

        # Query, Key, Value Matrix
        Q = X @ self.Qw
        K = X @ self.Kw
        V = X @ self.Vw

        # Batch_Size should be the same for Query, Key, and Value Matrix
        batch_size = Q.size(0)

        # Creates the key dimensions
        d_keys = d_model // self.heads

        Q = Q.view(batch_size, -1, self.heads, d_keys).permute(0,2,1,3)
        K = K.view(batch_size, -1, self.heads, d_keys).permute(0,2,1,3)
        V = V.view(batch_size, -1, self.heads, d_keys).permute(0,2,1,3)

        scaled_dot_prod = (Q @ K.permute(0,2,1)) / math.sqrt(d_keys)
        attention_prob = scaled_dot_prod.softmax(dim=-1)

        
        A = self.drop_out(attention_prob @ V)
        A = A.permute(0,2,1,3).view(batch_size, -1, self.heads * d_keys)
        
        attention_scores = self.Ow(A)
        
        return attention_scores

class LayerNorm(nn.Module):
    def __init__(self, shape:list, eps:int=1e-5):
        super().__init__()

        self.shape = shape

        # Declaring the Gamma Weights (Initiliazed with ones)
        self.G = nn.Parameter(torch.ones(shape))

        # Declaring the Beta Weights (Initialized with ones)
        self.B = nn.Parameter(torch.zeros(shape))

        # Declaring epsilon (small constant to prevent division of 0)
        self.epsilon = eps

    def forward(self, X):

		# Finding the mean of the input X
        X_m = X.mean(dim=-1, keepdims=True)

		# Finding the 
        X_v = X.var(dim=-1, keepdims=True, unbiased=True)

        return self.G * (X - X_m) / (torch.sqrt(X_v + self.epsilon)) + self.B 
        
