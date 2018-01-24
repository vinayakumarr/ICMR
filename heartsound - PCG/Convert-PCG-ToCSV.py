import pandas as pd
import csv
with open('1.csv', 'rb') as f:
    reader = csv.reader(f)
    your_list = list(reader)
