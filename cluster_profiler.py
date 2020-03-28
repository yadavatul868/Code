## Importing Libraries
import numpy as np
import pandas as pd
import re
import os
import inspect
import time
import datetime
import pandas as pd
import numpy as np
import datetime
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection,ensemble,metrics
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from IPython.display import display


class ClusterProfiler():
	
	def __init__(self, data, cluster_column, percentage_dist = True):
		
		self.data = data
		self.cluster_column = cluster_column
		self.percentage_dist = percentage_dist
		
		
		
	def numericalProfiler(self, numerical_column_list):
		
		self.numerical_column_list = numerical_column_list
	
		columns_list = self.numerical_column_list + [self.cluster_column]
		data_subset = self.data[columns_list]
		df_stat_main = pd.DataFrame()
		for i in data_subset[self.cluster_column].unique().tolist():
			df_temp = data_subset[data_subset[self.cluster_column] == i][self.numerical_column_list]
			df_stat = df_temp.describe().T.reset_index().rename(columns = {'index':'Features'})
			df_stat['sum'] = df_temp.sum().values
			df_stat['Cluster'] = 'Cluster_' + str(int(i))
			df_stat_main = df_stat_main.append(df_stat, ignore_index = True)
		return df_stat_main
		

		
	def categoricalProfiler(self,categorical_column_list):
		
		self.categorical_column_list = categorical_column_list
		columns_list = self.categorical_column_list + [self.cluster_column]
		data_subset = self.data[columns_list]
		

		# overall feature creation
		df_main_overall = pd.DataFrame()
		for i in data_subset.columns.tolist():
			if i not in [self.cluster_column]:
				df_temp = (data_subset.groupby(i).size().to_frame().reset_index().rename(columns = {0:'Overall',i:'Features'}))
				df_temp['prefix'] = str(i) + '_'
				if self.percentage_dist == True:
					df_temp['suffix'] = ' (% Dist)'
				else:
					df_temp['suffix'] = ' (Abs Dist)'
				df_temp['Features'] = df_temp['prefix'] + df_temp['Features'].astype(str) + df_temp['suffix']
				df_temp.drop(['prefix','suffix'], axis = 1, inplace = True)
				df_main_overall = df_main_overall.append(df_temp)
		df_part1 = df_main_overall.reset_index(drop = True)

		# cluster level feature creation
		df_main_cluster = pd.DataFrame()
		for i in data_subset.columns.tolist():
			if i not in [self.cluster_column]:
				df_temp = (data_subset.groupby([i,self.cluster_column]).size().to_frame().reset_index().rename(columns = {0:'Cluster_count',i:'Features'}))
				df_temp['prefix'] = str(i) + '_'
				if self.percentage_dist == True:
					df_temp['suffix'] = ' (% Dist)'
				else:
					df_temp['suffix'] = ' (Abs Dist)'
				df_temp['Features'] = df_temp['prefix'] + df_temp['Features'].astype(str) + df_temp['suffix']
				df_temp.drop(['prefix','suffix'], axis = 1, inplace = True)
				df_main_cluster = df_main_cluster.append(df_temp)
		df_main_cluster[self.cluster_column] = (df_main_cluster[self.cluster_column].astype(int)).astype(str)
		df_part2 = pd.pivot_table(df_main_cluster, values = 'Cluster_count',
								  index=['Features'], columns = self.cluster_column).reset_index().fillna(0)
		
		# adding prefix to the column names
		keep_same = {'Features', 'Overall'}
		df_part2.columns = ['{}{}'.format('' if c in keep_same else 'Cluster_', c) for c in df_part2.columns]
		
		# merge both datasets
		data_merged_overall = pd.merge(df_part2 , df_part1 , right_on = 'Features' , left_on = 'Features' , how  = 'left')
		
		# converting numbers to percentages
		if self.percentage_dist == True:
			cluster_cols = [j for j in data_merged_overall.columns.tolist() if j not in keep_same]
			for i in cluster_cols:
				data_merged_overall[i] = (data_merged_overall[i]/data_merged_overall['Overall']*100).round(2)
		
		return data_merged_overall







