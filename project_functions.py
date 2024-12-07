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


class model_metrics:
    def __init__(self, model, y_test, y_pred):
        self.model = model
        self.y_test = y_test
        self.y_pred = y_pred
        
    def get_summary(self):
        # building the confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        # Print the model metrics and confusion matrix
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

        
        
class univariate_visualization:
    def __init__(self, df, numerical_columns, cols=3, bins=20):
        self.df = df
        self.numerical_columns = numerical_columns
        self.cols = cols
        self.bins = bins
        
    def visualize(self):
        num_columns = len(self.numerical_columns)
        rows = math.ceil(num_columns / self.cols)
        
        # Create subplots
        fig, axes = plt.subplots(rows, self.cols, figsize=(6 * self.cols, 4 * rows), constrained_layout=True)
        # flattening for easier iteration
        axes = axes.flatten()
        
        for i, column in enumerate(self.numerical_columns):
            sns.distplot(self.df[column], bins=self.bins, ax=axes[i], kde=True, hist=True)
            axes[i].set_title(f'Distribution of {column}')
            axes[i].set_xlabel(column)
            axes[i].set_ylabel('Density')
            
        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

## creating a class function that will visualize the categorical columns against the target column            
class bivariate_visualization:
    
    def __init__(self, df, categorical_columns, target_column):
        self.df = df
        self.categorical_columns = categorical_columns
        self.target_column = target_column
    
    def visualize(self):
        for column in self.categorical_columns:
            plt.figure(figsize=(8, 5))
            sns.countplot(x=column, hue=self.target_column, data=self.df, palette='Set2')
            plt.title(f'Distribution of {column} by {self.target_column}')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.legend(title=self.target_column, loc='upper right')
            plt.show()
        
        
        
class roc_curve_plotter:
    def __init__(self, y_test):
        self.y_test = y_test
        self.models = []

    def add_model(self, y_pred, model_name):
        """
        fpr= false positive rate
        tpr= true positive rate
        """
        fpr, tpr, _ = roc_curve(self.y_test, y_pred)
        auc_score = auc(fpr, tpr)
        self.models.append((fpr, tpr, auc_score, model_name))

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
        
        

class correlation:
    def __init__(self, df):
        self.df = df
        
    def plot_correlationMatrix(self):
        corr_mat = self.df.corr()
        plt.subplots(figsize=(15,12))
        sns.heatmap(corr_mat, annot=True, cmap='PuBu');
        plt.xticks(rotation=90);
        plt.yticks(rotation=0);

