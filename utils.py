import os
import time
from torch.utils.tensorboard import SummaryWriter
import logging

def setattr_cls_from_kwargs(cls, kwargs):
    #if default values are in the cls,
    #overlap the value by kwargs
    for key in kwargs.keys():
        if hasattr(cls, key):
            print(f"{key} in {cls} is overlapped by kwargs: {getattr(cls,key)} -> {kwargs[key]}")
        setattr(cls, key, kwargs[key])

        
def test_setattr_cls_from_kwargs():
    class _test_cls:
        def __init__(self):
            self.a = 1
            self.b = 'hello'
    test_cls = _test_cls()
    config = {'a': 3, 'b': 'change_hello', 'c':5}
    setattr_cls_from_kwargs(test_cls, config)
    for key in config.keys():
        print(f"{key}:\t {getattr(test_cls, key)}")
        
        
def net_builder(net_name, net_conf=None, num_classes=10):
    """
    return **class** of backbone network (not instance).
    Args
        net_name: 'WideResNet' or network names in torchvision.models
        from_name: If True, net_buidler takes models in torch.vision models. Then, net_conf is ignored.
        net_conf: When from_name is False, net_conf is the configuration of backbone network (now, only WRN is supported).
    """
    import torchvision.models as models_torch
    builder = None
    if net_name == 'wrn':
        import models.nets.wrn as net
        builder = getattr(net, 'build_WideResNet')()
        setattr_cls_from_kwargs(builder, net_conf)
    elif net_name == 'cnn13':
        import models.nets.cnn13 as net
        builder = getattr(net, 'build_CNN13')()
    elif net_name=='resnet18':
        import models.nets.resnet18 as net
        base_net = models_torch.__dict__[net_name]
        builder = getattr(net, 'build_ResNet18')()
        setattr_cls_from_kwargs(builder, {'base_net':base_net})
    elif net_name == 'preresnet':
        import models.nets.prern as net
        builder = getattr(net, 'build_prern')()
    else:
        assert Exception("Not Implemented Error")
            
    return builder.build

    
def test_net_builder(net_name, from_name, net_conf=None):
    builder = net_builder(net_name, from_name, net_conf)
    print(f"net_name: {net_name}, from_name: {from_name}, net_conf: {net_conf}")
    print(builder)

    
def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    log_format = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)
    
    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    
    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
