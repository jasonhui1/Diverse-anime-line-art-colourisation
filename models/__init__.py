from .standard_adain import *


def get_model(config):
    return globals()[config['arch']](**config['kwargs'])
