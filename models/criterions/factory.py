from bootstrap.lib.options import Options
from .cross_entropy import CrossEntropyLoss

def factory(engine, mode):
    name = Options()['model.criterion.name']
    split = engine.dataset[mode].split

    if name == 'cross_entropy':
        if split == 'test':
            return None
        criterion = CrossEntropyLoss()
    else:
        raise ValueError(name)

    return criterion