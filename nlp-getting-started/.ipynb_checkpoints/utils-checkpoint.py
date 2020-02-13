import matplotlib.pyplot as plt
import seaborn as sns

import string 

import math
from scipy import stats

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

############# EDA - Text Preprocessing ###############
    
def count_numbers(tweet):
    """Counts the number of numbers in a tweet"""
    tweet = tweet.translate(str.maketrans('', '',string.punctuation)).lower().strip().split()
    numbers = []
    for item in tweet:
        try:
            numbers.append(int(item))
        except:
            pass
        
    return len(numbers)
    

############## EDA - Graphs ##################

def plot_column_distributions(col_list,train_df,figsize,null_hypothesis=True):
    """Plot the distributions of a list of columns with hue=target"""
    
    rows = math.ceil(len(col_list)/2)
    
    fig, ax = plt.subplots(nrows=rows,ncols=2,figsize=figsize)
    for col, ax in zip(col_list, ax.ravel()):
        target1 = train_df[col][train_df['target']==1]
        target0 = train_df[col][train_df['target']==0]
        sns.kdeplot(target1,ax=ax,label=1,shade=True)
        sns.kdeplot(target0,ax=ax,label=0,shade=True)

        ax.set_title('_'.join(col.split("_")[1:]))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
            
    plt.subplots_adjust(hspace=0.7)
    
    
    
############# MODEL EVALUATION ##########
def create_report(y_test,pred):
    """Create confusion matrix heatmap, classification report and print the overall accuracy
    
    Args:
        y_test (list): true target labels
        pred (list): predicted target labels
    """
    
    sns.heatmap(confusion_matrix(y_test, pred),cmap='Blues',annot=True,fmt='.0f',cbar=False)
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

    print(classification_report(y_test,pred))
    print(f"F1 score: {f1_score(y_test,pred)}")