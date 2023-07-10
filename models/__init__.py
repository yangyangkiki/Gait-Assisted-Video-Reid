from __future__ import absolute_import

from .ResNet import *
from .ResNet_Baseline import *

__factory = {
    'resnet50tp_2branch': ResNet50_2Branch_TP,
    'resnet50tp_2branch_kd': ResNet50_2Branch_TP_KD,
}

def get_names():
    return __factory.keys()

def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
