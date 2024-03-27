import torch

class Positional_Encoder:
    def __init__(self, max_length, batch):
        self.max_length = max_length
        self.batch = batch
        
    def fit_transform(self):
        d_model = len(self.batch)
        z_t = torch.zeros(2, self.max_length)
        
        pos_vector = torch.Tensor([[i] for i in range(d_model)])

        even_index = torch.arange(0, self.max_length, 2)
        odd_index = torch.arange(1, self.max_length, 2)

        even_denominator = torch.pow(10000, (2 * even_index) / d_model)
        odd_denominator = torch.pow(10000, (2 * odd_index) / d_model)

        even_pos_encoding = torch.sin(pos_vector / even_denominator)
        odd_pos_encoding = torch.cos(pos_vector / odd_denominator)


        j = k = 0
        for i in range(self.max_length):
            if i % 2 == 0:
                z_t[:, i] = even_pos_encoding[:, j]
                j += 1
            else:
                z_t[:, i] = odd_pos_encoding[:, k]
                k += 1

        return z_t