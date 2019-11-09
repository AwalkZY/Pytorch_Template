import logging
import os
import time

import torch

from utils.accessor import create_file


class Debugger:
    def __init__(self, filename=None):
        time_string = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        filename = time_string if filename is None else filename
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        if not os.path.exists("log"):
            os.mkdir("log")
        filename = "log/" + filename + ".log"
        create_file(filename)
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        self.logger.addHandler(console)

    def info(self, *inputs):
        self.logger.info(inputs)

    def warning(self, *inputs):
        self.logger.warning(inputs)

    def debug(self, *inputs):
        self.logger.debug(inputs)

    def exception(self, *inputs):
        self.logger.exception(inputs)


if __name__ == "__main__":
    debugger = Debugger()
    a = torch.tensor([1, 2, 3])
    debugger.debug("Value of a:", a)
