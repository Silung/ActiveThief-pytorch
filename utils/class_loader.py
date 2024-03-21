def load_dataset(dataset, markable=False):
    ds = None
    if dataset == 'cifar':
        if markable: 
            from dataset.cifar_dataset import CifarMarkableDataset
            ds = CifarMarkableDataset
        else:
            from dataset.cifar_dataset import CifarDataset
            ds = CifarDataset
    elif dataset == 'imagenet':
        if markable:
            from dataset.imagenet_dataset import ImagenetMarkableDataset
            ds = ImagenetMarkableDataset
        else:
            from dataset.imagenet_dataset import ImagenetDataset
            ds = ImagenetDataset
    elif dataset == 'mnist':
        if markable:
            from dataset.mnist_dataset import MnistMarkableDataset
            ds = MnistMarkableDataset
        else:
            from dataset.mnist_dataset import MnistDataset
            ds = MnistDataset
    elif dataset == 'mnist_small':
        from dataset.mnist_dataset import MnistSmallDataset
        ds = MnistSmallDataset
    elif dataset == 'mnist_dist':
        from dataset.mnist_dataset import MnistDistillationDataset
        ds = MnistDistillationDataset
    elif dataset == 'usps':
        if markable:
            from dataset.usps_dataset import UspsMarkableDataset
            ds = UspsMarkableDataset
        else:
            from dataset.usps_dataset import UspsDataset
            ds = UspsDataset
    elif dataset == 'generated_cifar':
        if markable:
            from dataset.generated_dataset import GeneratedCifarMarkableDataset
            ds = GeneratedCifarMarkableDataset
        else:
            from dataset.generated_dataset import GeneratedCifarDataset
            ds = GeneratedCifarDataset
    elif dataset == 'generated_cifar_finetune':
        if markable:
            from dataset.generated_dataset import GeneratedCifarFinetuneMarkableDataset
            ds = GeneratedCifarFinetuneMarkableDataset
        else:
            from dataset.generated_dataset import GeneratedCifarFinetuneDataset
            ds = GeneratedCifarFinetuneDataset
    elif dataset == 'generated_mnist':
        if markable:
            from dataset.generated_dataset import GeneratedMnistMarkableDataset
            ds = GeneratedMnistMarkableDataset
        else:
            from dataset.generated_dataset import GeneratedMnistDataset
            ds = GeneratedMnistDataset
    elif dataset == 'generated_imagenet':
        if markable:
            from dataset.generated_dataset import GeneratedImagenetMarkableDataset
            ds = GeneratedImagenetMarkableDataset
        else:
            from dataset.generated_dataset import GeneratedImagenetDataset
            ds = GeneratedImagenetDataset
    elif dataset == 'random':
        if markable:
            from dataset.random_dataset import RandomMarkableDataset
            ds = RandomMarkableDataset
        else:
            from dataset.random_dataset import RandomDataset
            ds = RandomDataset
    elif dataset == 'ordered':
        if markable:
            from dataset.random_dataset import OrderedMarkableDataset
            ds = OrderedMarkableDataset
        else:
            from dataset.random_dataset import OrderedDataset
            ds = OrderedDataset
    elif dataset == 'gtsrb':
        if markable:
            from dataset.gtsrb_dataset import GtsrbMarkableDataset
            ds = GtsrbMarkableDataset
        else:
            from dataset.gtsrb_dataset import GtsrbDataset
            ds = GtsrbDataset
    elif dataset == 'imagenet_tiny':
        if markable:
            from dataset.imagenet_tiny_dataset import ImagenetTinyMarkableDataset
            ds = ImagenetTinyMarkableDataset
        else:
            from dataset.imagenet_tiny_dataset import ImagenetTinyDataset
            ds = ImagenetTinyDataset
    elif dataset == 'agnews':
        if markable:
            from dataset.agnews_dataset import AgnewsMarkableDataset
            ds = AgnewsMarkableDataset
        else:
            from dataset.agnews_dataset import AgnewsDataset
            ds = AgnewsDataset
    elif dataset == 'imdb':
        if markable:
            from dataset.imdb_dataset import ImdbMarkableDataset
            ds = ImdbMarkableDataset
        else:
            from dataset.imdb_dataset import ImdbDataset
            ds = ImdbDataset
    else:
        raise Exception("Dataset {} could not be loaded".format(dataset)) 
    
    return ds


def load_model(model_class):
    model = None
    
    if model_class.startswith('resnet'):
        if model_class == 'resnet_pretrained':
            from models.resnet import ResNet_Pretrained
            return ResNet_Pretrained
        else:
            from models.resnet import ResNet
            return ResNet
    elif model_class.startswith('cnn_'):
        from models.cnn import CNN
        blocks, convs_in_block = model_class.strip().split('_')[1:]
        blocks, convs_in_block = int(blocks), int(convs_in_block)
    
        class CNNWrapper(CNN):
            def __init__(self, *args, **kwargs):
                super(CNNWrapper, self).__init__(*args, convs_in_block=convs_in_block, num_filters=[32, 64, 128, 256][:blocks], **kwargs)

        return CNNWrapper
    elif model_class.startswith('net_'):
        from models.net import NET
        return NET
    
    else:
        raise Exception("Model {} could not be loaded" .format(model_class)) 
        
    return model
