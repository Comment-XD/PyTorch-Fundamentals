import torch
import torch.nn as nn

class Positional_Encoding(nn.Module):
    def __init__(self, drop_value:float=0.1, n:int=10000, d_model:int=4, max_length:int=5000):
        super().__init__()
        """
        drop_value (float) drop out value 
        n: (int) constant value 
        d_model (int): the dimension of the word embeddings 
        max_length (int): the total words in a sentence 
        
        """
        self.divisor = torch.exp(torch.arange(0, d_model, 2) * -(math.log(n) / d_model)) # Creates the denominator component 
        self.drop_out = nn.Dropout(drop_value)
    
        self.k = torch.arange(0, max_length).unsqueeze(dim=1) # Creates the position elements 
        self.pe = torch.zeros(max_length, d_model) # Creates the empty array that will store the positional encodings
        
        self.pe[:, 0::2] = torch.sin(self.k * self.divisor) # For every row and even column, implement sin function
        self.pe[:, 1::2] = torch.cos(self.k * self.divisor)
        
        self.pe.squeeze(0)

    def forward(self, x):
        return self.drop_out(x + self.pe[:x.size(0)].requires_grad_(False))
        
class Scalar_Product_Attention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        