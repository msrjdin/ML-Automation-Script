import pandas as pd

class DetectingColTypes:
    def __init__(self, df):
        print('Detecting ColTypes')

    def returnValues(self):
        colTypes = {'A': 'Identity', 'B': 'Categorical'}
        return colTypes