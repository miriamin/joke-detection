import pandas as pd
import csv
from sklearn.utils import shuffle


class JokePreprocessor:
    def load(self,file):
        prefix = file[-3:]
        if prefix == 'txt':
            outfile = file[:-3] + 'csv'
            with open(file) as file_, open(outfile, 'w') as csvfile:
                lines = [x for x in file_.read().strip().split('$$$$$') if x]
                writer = csv.writer(csvfile, delimiter='|')
                writer.writerow(('ID', 'Joke'))
                for idx, line in enumerate(lines, 1):
                    writer.writerow((idx, line.strip('$$$$$')))
        jokes_df = pd.read_csv(file[:-3]+'csv', delimiter='|', index_col='ID')
        return jokes_df

    def add_labels(self, dataframe, label):
        dataframe['Label'] = label
        return dataframe

    def merge_and_shuffle (self,dfs):
        full_df = pd.concat(dfs)
        full_df = shuffle(full_df)
        return full_df

