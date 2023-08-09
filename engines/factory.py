from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
from engines.engine import Engine
def factory():
    engine = Engine()
    return engine