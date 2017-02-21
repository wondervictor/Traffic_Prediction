from paddle.trainer.PyDataProvider2 import *
import numpy as np



def init_hook(settings, **kwargs):
    pass


@provider(init_hook=init_hook)
def process(settings, filename):
    pass
