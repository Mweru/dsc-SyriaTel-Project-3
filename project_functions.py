#importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_curve, confusion_matrix, auc
from sklearn.preprocessing import StandardScaler
import math
import warnings
warnings.filterwarnings('ignore')



#creating a class that will take the dataframe and relevant columns and visualize for both univariate visuals and bivariate visuals
class visualizations:
    def __init__(self, df, columns):
        self.df = df
        self.columns = columns
    
    #defining the univariate distribution plot function
    def visualize_univariate(self, cols=3, bins=20):
        
        num_columns = len(self.columns)
        rows = math.ceil(num_columns / cols)
        
        # Create subplots and flatten for easier iterations
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), constrained_layout=True)
        axes = axes.flatten()
        
        #Iterating through each column and visualizing it 
        for i, column in enumerate(self.columns):
            sns.distplot(self.df[column], bins=bins, ax=axes[i], kde=True, hist=True)
            axes[i].set_title(f'Distribution of {column}')
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Density')
            
    #function will visualize countplots for each column againt the target column
    def visualize_bivariate(self, target_column):
        for column in self.columns:
            plt.figure(figsize=(8, 5))
            sns.countplot(x=column, hue=target_column, data=self.df, palette='Set2')
            plt.title(f'Distribution of {column} by {target_column}')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.legend(title=target_column, loc='upper right')
            plt.show()
            
    #function will visualize a correlation matrix of the columns
    def visualize_multivariate(self):
        corr_mat = self.df.corr()
        plt.subplots(figsize=(15,12))
        sns.heatmap(corr_mat, annot=True, cmap='PuBu');
        plt.xticks(rotation=90);
        plt.yticks(rotation=0);
        


#building a class that will print model performance summaries when given the model, the test sample and predicted sample
class model_metrics:
    def __init__(self, model, y_test, y_pred):
        self.model = model
        self.y_test = y_test
        self.y_pred = y_pred
        
    def get_summary(self):
        # building the confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Print the accuracy, f1, precision and recall scores
        print(f'{self.model} Model Metrics')
        print(f'______________________________')
        print(f'The accuracy score is {round(accuracy_score(self.y_test, self.y_pred), 3)}.')
        print(f'The F1 score is {round(f1_score(self.y_test, self.y_pred), 3)}.')
        print(f'The recall score is {round(recall_score(self.y_test, self.y_pred), 3)}.')
        print(f'The precision score is {round(precision_score(self.y_test, self.y_pred), 3)}.')
        print(f'______________________________')
        
        # Plot the confusion matrix
        sns.heatmap(cm, annot=True, fmt='d')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix');

        
#creating a class that will plot ROC curves for the models given the test values       
class roc_curve_plotter:
    def __init__(self, y_test):
        self.y_test = y_test
        self.models = []

    #function will take the model name, and predicted values
    def add_model(self, y_pred, model_name):
  
        #fpr= false positive rate
        #tpr= true positive rate
        fpr, tpr, _ = roc_curve(self.y_test, y_pred)
        auc_score = auc(fpr, tpr)
        #adds the model to a list of models
        self.models.append((fpr, tpr, auc_score, model_name))

    #function plots each model's ROC curve in one figure
    def plot(self, figsize=(10, 8)):
        plt.figure(figsize=figsize)

        # Plot each model's ROC curve
        for fpr, tpr, auc_score, model_name in self.models:
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")
        
        # Diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.50)")

        # Add labels and title
        plt.title("ROC Curves for Multiple Models")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
        
        

