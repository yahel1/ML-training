import pandas as pd
import glob

import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO 
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
from IPython.display import Image, display
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from collections import defaultdict
from tqdm.autonotebook import tqdm

import seaborn as sns
import scipy.special
from scipy import stats
import matplotlib.pyplot as plt
import itertools
import holoviews as hv
hv.extension('bokeh')
# from bokeh.layouts import gridplot
from bokeh.plotting import figure, show

import pickle as pkl
import pydotplus
import os

# This is a bit of magic to make matplotlib figures appear inline in the
# notebook rather than in a new window.
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def plot_distribution(X_trues, X_falses):
    """
    Plot the distribution of values a single feature across two classes.
        Example:
            - Feature: height
            - Class: is_psagotnik
    
    """
    a=hv.Distribution(X_trues, label='True')
    b=hv.Distribution(X_falses, label='False')
    display((a*b).options(width=800, height=400))


def plot_binary_distribution(X_trues, X_falses):
    """
    Plot the distribution of values a single feature across two classes for binary (0, 1) features.
        Example:
            - Feature: did call Hana last week
            - Class: is_psagotnik

    """
    assert len([x for x in X_trues if x == 1 or x==0]) == len(X_trues), "X_trues should have only 1, 0"
    assert len([x for x in X_falses if x == 1 or x==0]) == len(X_falses), "X_falses should have only 1, 0"

    true_1_density = len([x for x in X_trues if x == 1])/len(X_trues)
    true_0_density = len([x for x in X_trues if x == 0])/len(X_trues)

    false_1_density = len([x for x in X_falses if x == 1])/len(X_falses)
    false_0_density = len([x for x in X_falses if x == 0])/len(X_falses)

    index, groups = ['Trues', 'Falses'], ['1', '0']
    keys = itertools.product(index, groups)
    values = [true_1_density, true_0_density, false_1_density, false_0_density]
    bars = hv.Bars([k+(v,) for k, v in zip(keys, values)],
                   ['Index', 'Group'], '%')
    print(values)
    stacked = bars.opts(stacked=True, clone=True)
    display(bars.relabel(group='Grouped') + stacked.relabel(group='Stacked'))


def t_test(arr1, arr2):
    """
    Calculate the T-test for the means of *two independent* samples of scores.

    Test if arr1 and arr2 come from the same distribution.
    This can help us test if for a single feature f1 behaves differently for two different populations (i.e. boys, girls)

    Returns
    -------
    pvalue : float or array
        The two-tailed p-value (smaller means that we have a higher chance that the two populations behave differently).
            lower than 0.05 usually means that they ARE DIFFERENT

    Note:
        You can refer to the documentation of stats.ttest_ind() for more details
    """
    return stats.ttest_ind(arr1, arr2)[1]


# holoviews confusion
def plot_confusion_matrix(y_pred, 
                        y_true):
    """
        Plots an interactive confusion matrix using 
    """
    pdf = pd.DataFrame(list(zip(y_pred, y_true)), columns=['Prediction', 'Actual'])

    graph = pdf.groupby(['Prediction', 'Actual']).size().to_frame().reset_index()
    confusion = graph.rename(columns={0: 'Count'})
    # in a format for holoviews
    conf_values = map(lambda l: [str(l[0]), str(l[1]), l[2]], [a.tolist() for a in confusion.values])  
    p = hv.HeatMap(conf_values, label='Confusion Matrix', kdims=['Predicted', 'Actual'], vdims=['Count']).sort().options(
            xrotation=45, width=400, height=400, cmap='blues', tools=['hover'], invert_yaxis=True, zlim=(0,1))
    display(p)
    
    
def plot_descision_tree(tree, train_df):
    """
    T - terrorist
    NT - not terrorist
    View the resulting graph to see your trained descision tree
    """
    dot_data = StringIO()
    export_graphviz(tree, out_file=dot_data, feature_names=train_df.columns, class_names=['NT', 'T'],
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("dtree.png")
    # from bokeh import figure 
    from bokeh.models.tools import WheelZoomTool
    wheel_zoom = WheelZoomTool()
    p = figure(width=900, height=400, tools=["pan",wheel_zoom,"reset"])
    p.image_url(url=['dtree.png'], x=0, y=0, w=5, h=1, anchor="bottom_left")
    p.toolbar.active_scroll = wheel_zoom
    show(p)
