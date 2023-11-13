class MarkableDataset():
    def __init__(self):
        self.state = 'unmark'
        
        # Initialize hashsets and hashmap
        self.marked = []
        self.marking = []
        self.unmark = list(range(len(self.data)))
        
        self.aux_data = {}
        self._marked_counter = 0
        self._unmark_counter = 0
        
    def set_state(self, state):
        self.state = state
        
    def __getitem__(self, index):
        if self.state == 'marked':
            idx = self.marked[index]
        elif self.state == 'marking':
            idx = self.marking[index]
        else:
            idx = self.unmark[index]
        x = self.data[idx]
        y = self.labels[idx]
        
        if idx in self.aux_data:
            aux = self.aux_data[idx]
        else:
            aux = 0
        return x, y, idx, aux
        
    def __len__(self):
        if self.state == 'marked':
            return len(self.marked)
        elif self.state == 'marking':
            return len(self.marking)
        else:
            return len(self.unmark)
        
    def mark(self, i):
        assert i not in self.marked
        
        if i not in self.marked:
            self.unmark.remove(i)
            self.marking.append(i)
        
    def update(self, i, aux_data=None):
        self.marking.remove(i)
        self.marked.append(i)
        self.aux_data[i] = aux_data