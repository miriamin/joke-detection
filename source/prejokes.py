import pandas as pd
import csv
from sklearn.utils import shuffle
import re
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import spacy

from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity




def remove_punct_from_text(text):
    text_nopunct = ''
    text_nopunct = re.sub('[' + string.punctuation + ']', '', text)
    return text_nopunct

def lower_case_tokens(tokens):
    return [w.lower() for w in tokens]

def remove_stopwords(tokens):
    stoplist = stopwords.words('english')
    return [word for word in tokens if word not in stoplist]

def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{1,}', '##', x)
    return x



class JokePreprocessor:

    ### Data Gathering ###
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

    def get_max_seq_len(seld, df, col_name):
        seqs = df[col_name].tolist()
        tokenized = [word_tokenize(x) for x in seqs]
        seqs_lengths = [len(x) for x in tokenized]
        return max(seqs_lengths)

    ### Data Cleaning ###

    def remove_punctuation(self, df, col_name):
        df[col_name] = df[col_name].apply(lambda x: remove_punct_from_text(x))
        return df

    def replace_numbers(self, df, col_name):
        df[col_name] = df[col_name].apply(lambda x: clean_numbers(x))
        return df


    def clean_and_tokenize(self, df, col_name):
        df[col_name] = df[col_name].apply(lambda x: remove_punct_from_text(x))
        df['Tokens'] = df[col_name].apply(lambda x: word_tokenize(x))
        df['Tokens'] = df['Tokens'].apply(lambda x: lower_case_tokens(x))
        df['Tokens'] = df['Tokens'].apply(lambda x: remove_stopwords(x))
        #df = df[['Tokens', 'Label']]
        return df

    def one_hot_encode_label(self, df):
        df.loc[df['Label'] == 0, 'NoJoke'] = 1
        df.loc[df['Label'] == 0, 'Joke'] = 0
        df.loc[df['Label'] == 1, 'NoJoke'] = 0
        df.loc[df['Label'] == 1, 'Joke'] = 1
        df = df[['Tokens', 'NoJoke', 'Joke']]
        return df

    def count_sentences(self,text):
        sentences = sent_tokenize(text)
        return len(sentences)

    def sentence_n_grams_split(self, df, col_name, n):
        n_grams = []
        for i in df.index:
            sentence = df[col_name][i]
            sent_tokens = sent_tokenize(sentence)
            for j in range(len(sent_tokens)):
                if j + n <= len(sent_tokens):
                    n_gram_list = sent_tokens[j:j + n]
                    s= " "
                    n_gram = s.join(n_gram_list)
                    n_grams.append(n_gram)
        ngram_df = pd.DataFrame()
        ngram_df["joke_ngram"]=n_grams
        return ngram_df

    ### Prepare for Training ###

    def create_vocab(self, df):
        all_words = [word for tokens in df["Tokens"] for word in tokens]
        vocab = sorted(list(set(all_words)))
        return vocab

    def filter_by_joke_len(self, df, col_name, len):
        filtered = df.loc[df[col_name] == len]
        return filtered

    ### calculalte similarities ###
    def similarity_matrix(self, joke_df, plot_df):
        joke_df.reset_index()
        for i in range(len(joke_df.index)):
            col_name = 'plot_sentence_' + i

            # use sklearn cosine_similarity





