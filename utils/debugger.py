import logging
import os
import time

import torch

from utils.accessor import create_file


class Debugger:
    def __init__(self):
        time_string = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(level=logging.INFO)
        filename = "log/" + time_string + ".log"
        create_file(filename)
        handler = logging.FileHandler(filename)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        self.logger.addHandler(console)

    def debug(self, *inputs, logging_level="INFO"):
        if logging_level == "INFO":
            self.logger.info(inputs)
        elif logging_level == "DEBUG":
            self.logger.debug(inputs)
        else:
            self.logger.warning(inputs)


if __name__ == "__main__":
    debugger = Debugger()
    a = torch.tensor([1, 2, 3])
    debugger.debug("Value of a:", a, logging_level="INFO")
