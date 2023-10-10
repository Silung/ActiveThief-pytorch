def load_dataset(dataset):
    dsl = None

    if dataset == 'mnist':
        from dataset.mnist_dataset import MnistDataset
        dsl = MnistDataset

    elif dataset == 'cifar':
        from dataset.cifar_dataset import CifarDataset
        dsl = CifarDataset

    elif dataset == 'imagenet':
        from dataset.imagenet_dataset import ImagenetDataset
        dsl = ImagenetDataset
    
    elif dataset == 'uniform':
        from dataset.uniform_dataset import UniformDataset
        dsl = UniformDataset
    
    else:
        raise Exception("Dataset {} could not be loaded".format(dataset)) 
    
    return dsl

def load_noise_dataset(dataset):
    ds = None
    if dataset == 'cifar':
        from dataset.cifar_dataset import CifarNoiseDataset
        ds = CifarNoiseDataset
    elif dataset == 'imagenet':
        from dataset.imagenet_dataset import ImagenetNoiseDataset
        ds = ImagenetNoiseDataset
    elif dataset == 'mnist_small':
        from dataset.mnist_dataset import MnistSmallDataset
        ds = MnistSmallDataset
    else:
        raise Exception("Dataset {} could not be loaded".format(dataset)) 
    
    return ds


def load_model(model_class):
    model = None
    
    if model_class.startswith('cnn_'):
        from models.cnn import CNN
        blocks, convs_in_block = model_class.strip().split('_')[1:]
        blocks, convs_in_block = int(blocks), int(convs_in_block)
    
        class CNNWrapper(CNN):
            def __init__(self, *args, **kwargs):
                super(CNNWrapper, self).__init__(*args, convs_in_block=convs_in_block, num_filters=[32, 64, 128, 256][:blocks], **kwargs)

        return CNNWrapper
    
    else:
        raise Exception("Model {} could not be loaded" .format(model_class)) 
        
    return model
