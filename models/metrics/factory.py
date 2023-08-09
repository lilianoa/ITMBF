from bootstrap.lib.options import Options
from .accuracy import Accuracy
from .metric_test import Metric_test
def factory(engine, mode):
    name = Options()['model.metric.name']
    metric = None
    if name == 'accuracy':
        metric = Accuracy()
    elif name == 'metric_test':
        metric = Metric_test()
    else:
        raise ValueError(name)
    return metric