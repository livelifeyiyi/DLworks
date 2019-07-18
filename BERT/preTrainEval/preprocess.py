import pandas as pd
import codecs
import json

import re
from tqdm import tqdm
import ast

file_names_jsonl = ['skill_word_to_entity.jsonl']
all_words = []


def add_words(word, all_words):
	# / , \u300, \n
	if isinstance(word, str):
		word = word.replace(' ', '').replace('\n', '').replace('\\u300', '/')
		word = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", word)
		word = re.sub(u"\\（.*?）|\\{.*?}|\\[.*?]|\\【.*?】", "", word.decode())
		word = re.sub(r"\([（^)]）*\)", "", word)
		if '/' in word or ',' in word or '、' in word:
			all_words += re.split(r'[/,、]', word)
		else:
			all_words.append(word)
	elif isinstance(word, list):
		for each in word:
			each = each.replace(' ', '').replace('\n', '').replace('\\u300', '/')
			word = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", word)
			word = re.sub(u"\\（.*?）|\\{.*?}|\\[.*?]|\\【.*?】", "", word.decode())
			if '/' in each or ',' in each or '、' in each:
				all_words += re.split(r'[/,、]', each)
			else:
				all_words.append(word)
		# all_words += word
	else:
		print(word)
	return all_words


for file_name in file_names_jsonl:
	with codecs.open(file_name, encoding='utf-8', mode='r') as infile:
		for line_num, each_line in tqdm(enumerate(infile.readlines()), desc='processing file %s' % file_name):
			word = ast.literal_eval(each_line)[0]
			all_words.append(word)

df_degree = pd.read_csv("degrees.csv", header=0)
for each in df_degree["name"]:
	all_words = add_words(each, all_words)
# all_words += list(df_degree["name"])
del df_degree

df_departments = pd.read_csv("departments.csv", header=0)
for each in df_departments['name']:
	all_words = add_words(each, all_words)

for each in df_departments['aliases']:
	all_words = add_words(each, all_words)
	# if isinstance(each, str):
	# 	all_words += each.split(',')
del df_departments

df_functions = pd.read_csv('functions_cluster.csv', header=0)
for each in df_functions['name']:
	all_words = add_words(each, all_words)
del df_functions

df_unction = pd.read_csv('function_taxonomy.txt', header=0, sep='\t')
for each in df_unction['name']:
	all_words = add_words(each, all_words)
del df_unction

df_gsystem = pd.read_csv('gsystem_industries.txt', header=0, sep='\t')
for each in df_gsystem['name']:
	all_words = add_words(each, all_words)
del df_gsystem

df_industries = pd.read_csv('industries.csv', header=0)
for each in df_industries['name']:
	all_words = add_words(each, all_words)
del df_industries

csv_filenames = ['majors.csv', 'region.csv', 'schools.csv', 'skill_certificates.csv']
for csv_name in csv_filenames:
	df = pd.read_csv(csv_name, header=0)
	for each in df['name']:
		all_words = add_words(each, all_words)

# with codecs.open('skill_word_to_entity.jsonl', encoding='utf-8', mode='r') as infile:
# 	for line_num, each_line in tqdm(enumerate(infile.readlines()), desc='processing file %s' % file_name):
# 		all_words += list(json.loads(each_line)[0].keys())

all_words = list(set(all_words))
with codecs.open('dictionary.txt', encoding='utf-8', mode='a+') as outfile:
	for word in all_words:
		outfile.write(word + '\n')