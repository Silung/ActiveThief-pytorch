import os
import torch

# true model
true_model_dir = os.path.join('saved/cnn_3_2/mnist/true/trained_model.pth')
source_model_type = load_model('cnn_3_2')
true_model = source_model_type(1, 10, 'mnist')
true_model.load_state_dict(torch.load(os.path.join(true_model_dir, 'trained_model.pth')))
true_model = true_model.to(args.device)