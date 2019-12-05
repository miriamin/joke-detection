import pandas as pd
import csv

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

