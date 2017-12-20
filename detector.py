class Detector(object):
    def __init__(self, trainer, reader, args):
        self.trainer = trainer
        self.reader = reader
        self.args = args

    def process(self, out_dir):
        pass