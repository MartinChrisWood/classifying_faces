""" --------------------------------------------------------------------
                Import (a vast number of) modules                     
---------------------------------------------------------------------"""

import os
import pickle

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import tree

from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils import check_random_state
from sklearn.tree import _tree
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot as plt


""" Import some decision tree plotting functions I stole from a
blog post on the internet """
import decision_tree_plotting_functions as treePlot

""" The below for plotting a correlation matrix... """
from pylab import pcolor, show, colorbar, xticks, yticks

""" --------------------------------------------------------------------
		Load & clean data   
      Select columns of interest as features, and labels         
---------------------------------------------------------------------"""


data = pd.read_csv("./data/collated_results.csv")

features_list = ['name_in_url_BOOL',
				'name_BOOL',
				'postcode_BOOL',
				'full_address_BOOL',
				'partial_name_present',
				'partial_address_present',
				'partial_name_in_url',
				'links_ratio',
				'pagecounter',
				'rank']

labels_list = ['correct_BOOL']


x_data = pd.DataFrame(data, columns = features_list)
		
for each in features_list[0:4]:
	x_data[each] = x_data[each].apply(np.float)


y_data = pd.DataFrame(data, columns = labels_list)

for each in labels_list:
	y_data[each] = y_data[each].apply(np.float)

y_data = y_data.values.ravel()



""" --------------------------------------------------------------------
		TEST - try building a really basic tree, score it
-------------------------------------------------------------------- """

tree_model = tree.DecisionTreeClassifier(criterion='entropy')

tree_model.fit(x_data, y_data)

print(tree_model.score(x_data, y_data))

print(tree_model)

print(tree_model.tree_)



""" --------------------------------------------------------------------
		TEST - Plot up the end result
-------------------------------------------------------------------- """

treePlot.plot_tree_structure(tree_model, 
							feature_names=features_list,
							class_names=["False", "True"],
							savefile="./figures/test_tree.png")

treePlot.plot_confusion_matrix(y_data['correct_BOOL'].values, tree_model.predict(x_data),
						classes=["False", "True"],
						title = "Decision Tree Classifications Summary",
						savefile="./figures/test_tree_matrix.png")
						
						

""" --------------------------------------------------------------------
		TEST - Let's try a random forest...
-------------------------------------------------------------------- """

forest_model = RandomForestClassifier(n_estimators=3000,
										criterion="entropy",
										oob_score=True)

forest_model.fit(x_data, y_data)

print(forest_model.score(x_data, y_data))

print(forest_model)

print(forest_model.estimators_[0])


print("Out-Of-Bag error:  %f" % float(1.0 - forest_model.oob_score_))



""" --------------------------------------------------------------------
		TEST - Plot up the end result
-------------------------------------------------------------------- """

treePlot.draw_tree(forest_model, savefile="./figures/forest_stats.png")
print("plotted forest stats")
treePlot.draw_ensemble(forest_model, savefile="./figures/forest_ensemble.png")
print("plotted ensemble distribution")
treePlot.plot_confusion_matrix(y_data['correct_BOOL'].values, forest_model.predict(x_data),
						classes=["False", "True"],
						title = "Random Forest Classifications Summary",
						savefile="./figures/forest_matrix.png")
print("plotted forest confusion matrix")

