from prettytable import PrettyTable
import logging
import os
import sys

class Logger():
    def __init__(self, model_path, hyperparameters):
        self.model_path = model_path
        self.hyperparameters = hyperparameters
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        #logging settings
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)
        output_file_handler = logging.FileHandler(model_path + 'log')
        #stdout_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(output_file_handler)
        #self.logger.addHandler(stdout_handler)
        #log the hyperparameters
        #items = [i for i in self.hyperparameters.keys()]
        #values = [i for i in self.hyperparameters.values()]
        #x = PrettyTable()
        #x.add_column('items', items)
        #x.add_column('values', values)
        self.logger.info(self.hyperparameters)

    def get_logger(self):
        return self.logger

