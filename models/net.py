from torch import ones, cumsum, arange
from torch import nn

class NET(nn.Module):
    def __init__(self, num_class, dataset_name, vocab_size, hidden_dim=12, embed_dim=12, act1='SELU'):
        super().__init__()
        
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)  
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act1 = nn.__getattribute__(act1)()
        
        self.fc2 = nn.Linear(hidden_dim, num_class)
        #self.act2 = nn.__getattribute__(act2)()
        
        self.logs = nn.LogSoftmax(dim=-1)
        
        self.dropo = nn.Dropout(0.25)
        
    def count_weights_biases(self):
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, text, offsets):
        res = self.embedding(text, offsets)
        res = self.act1(self.dropo(self.fc1(res)))
        res = self.dropo(self.fc2(res))
        res = self.logs(res)
        return res