import pandas as pd
import codecs
import json
from tqdm import tqdm
import ast

file_names_jsonl = []
all_words = []

for file_name in file_names_jsonl:
	with codecs.open(file_name, encoding='utf-8', mode='r') as infile:
		for line_num, each_line in tqdm(enumerate(infile.readlines()), desc='processing file %s' % file_name):
			word = ast.literal_eval(each_line)[0]
			all_words.append(word)

df_degree = pd.read_csv("degree.csv", header=0)
all_words += list(df_degree["name"])
del df_degree

df_departments = pd.read_csv("departments.csv", header=0)
all_words += list(df_departments['name'])

for each in df_departments['aliases']:
	all_words += each.split(',')
del df_departments

df_functions = pd.read_csv('functions_cluster.csv', header=0)
all_words += list(df_functions['name'])
del df_functions

df_unction = pd.read_csv('unction_taxonomy.txt', header=0, sep='\t')
all_words += list(df_unction['name'])
del df_unction

df_gsystem = pd.read_csv('gsystem_industries.txt', header=0, sep='\t')
all_words += [each.split('/') for each in list(df_gsystem['name'])]
del df_gsystem

df_industries = pd.read_csv('industries.csv', header=0)
all_words += [each.split('/') for each in list(df_industries['name'])]
del df_industries

csv_filenames = ['majors.csv', 'region.csv', 'schools.csv', 'more_skill_certificates.csv']
for csv_name in csv_filenames:
	df = pd.read_csv(csv_name, header=0)
	all_words += list(df['name'])

with codecs.open('skill_word_to_entity.jsonl', encoding='utf-8', mode='r') as infile:
	for line_num, each_line in tqdm(enumerate(infile.readlines()), desc='processing file %s' % file_name):
		all_words += list(json.loads(each_line)[0].keys())

all_words = list(set(all_words))
with codecs.open('dictionary.txt', encoding='utf-8', mode='a+') as outfile:
	for word in all_words:
		outfile.write(word + '\n')