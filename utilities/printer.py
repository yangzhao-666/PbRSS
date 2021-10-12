import numpy as np
from prettytable import PrettyTable

class Printer():
    def __init__(self):
        return None

    def go(self, stuff):
        self.table = PrettyTable()
        items = [i for i in stuff.keys()]
        values = [i for i in stuff.values()]
        self.table.add_column('items', items)
        self.table.add_column('values', values)
        print(self.table, flush=True)
