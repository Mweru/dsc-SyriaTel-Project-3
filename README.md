## dsc-SyriaTel-Project-3
# Data Science end of Phase 3 Individual Project

## BUSINESS UNDERSTANDING
### Business Overview
Syriatel (Arabic: سيريتل) is a leading telecommunications company in Syria, known for its rapid growth and extensive market presence. With a robust network of 63 Points of Service across the country, Syriatel handles over 25,000 customer queries daily through its Call Centers and operates 2,783 radio base stations. The company proudly serves over 6 million customers, holding a 55% share of the Syrian market. Their skilled team is committed to delivering high-quality services and solutions, solidifying Syriatel’s position as one of the region's fastest-growing telecom operators.

### Problem Statement
As new customers begin using a product, each contributes to the growth rate of that product. However, over time, some customers may discontinue their usage or cancel their subscriptions for various reasons. Churn refers to the rate at which customers cancel or choose not to renew their subscriptions, and a high churn rate can significantly impacts revenue.

Syriatel has observed an increase in customer churn and is concerned about the financial losses associated with customers who discontinue their services prematurely. 

### Objectives
To Determine the features that serve as early indicators of customer churn.

To Analyze and identify the underlying reasons why customers discontinue their service.

To Build a Predictive Model that is capable of accurately predicting when a customer is likely to discontinue their service.

### Success Criteron
This analysis aims to:

Identify Key Features: Determine at least five key features that strongly correlate with customer churn, providing actionable insights for Syriatel to monitor and address customer dissatisfaction effectively.

Develop a Predictive Model: Build a classifier model that achieves: At least 90% accuracy in predicting customer churn. A minimum precision of 75%, ensuring the model minimizes false positives and provides reliable predictions.

Support Business Decision-Making: Enable Syriatel to use the identified features and model predictions to implement targeted retention strategies, reducing churn and mitigating revenue loss.

## DATA UNDERSTANDING

The Churn in Telecom's dataset was sourced from [Kaggle](https://www.kaggle.com/becksddf/churn-in-telecoms-dataset). The dataset has 3,333 rows and 21 columns. Each column represents a customer and the columns represent the customer details. 


## DATA PREPARATION & ANALYSIS

Upon checking the data for duplicates, missing values/ null values and there were none. 

Univariate analysis was performed by visualizing the distributions of our numerical columns using distribution plots. 
The analysis revealed that 12 out of the 15 numerical variables follow a normal distribution, which is advantageous for modeling purposes. The selected numerical columns include; account length, number of voicemail messages, total day minutes, total day calls, total day charge, total evening minutes, total evening calls, total evening charge, total night minutes, total night calls, total night charge, total international minutes, total international calls, total international charge, and customer service calls.

Bivariate analysis was performed by visualizing the relationship between our selected categorical variables: area code, international plan,  and voice mail plan, and the target variable- churn using count plots. 
The count plots reveal that customers with an international plan and those without a voicemail plan are more likely to churn, suggesting these features may influence customer retention. However, area code does not appear to have a significant impact on churn, as the distribution is relatively consistent across different codes. These insights highlight the importance of service plans in understanding customer churn.

To further prepare the data for analysis, we;
Dropped/Removed irrelevant columns such as area code, phone number, and state. 
Transformed the categorical columns into binary format using one-hot encoding, with adjustments made to avoid multicollinearity. 
Merged the total minutes, calls, and charges into single combined columns for each category, reducing the dataset's size and improving its efficiency for analysis.

Multivariate analysis was performed by visualizing the correlationam matrix for the features to address any multicollinearity and optimize the dataset for modeling. 
The results showed that most features exhibit little to no correlation, indicating they are largely independent. However, we observed strong correlations between voice mail plan_yes and number vmail messages, as well as between total charge and total minutes. To address this, we evaluated their correlations with churn and dropped the feature with the weaker correlation to ensure the dataset remains relevant and non-redundant.

## MODELING & EVALUATION
In this analysis, we built and evaluated several machine learning models to predict customer churn for Syriatel. 

We employed the following models in our analysis:
- Logistic Regression: A simple model for binary classification tasks. It provides a strong baseline, allowing us to measure performance improvements with more complex models.
- Modified Logistic Regression (with SMOTE and MinMax Scaling): To improve upon the baseline Logistic Regression model, we applied the SMOTE technique to address class imbalance and used MinMax scaling to standardize the feature range. This modification aimed to enhance the model's performance, especially in terms of handling imbalanced data.
- Decision Tree Classifier: A decision tree is a non-linear model that works well for problems with complex relationships between features. It is robust to outliers and doesn’t require feature scaling, which made it a great option for this analysis.
- Random Forest Classifier: This ensemble model is made up of multiple decision trees and is known for its high accuracy, robustness to overfitting, and ability to handle complex data.
  
The success of these models is evaluated based on the following criteria:

- Accuracy: A high accuracy score indicates that the model correctly classifies the majority of instances.
- Precision: This metric shows how many of the predicted churns were actual churns, minimizing false positives.
- Recall: Recall is important for identifying the actual churns, minimizing false negatives.
- F1 Score: A balanced measure of precision and recall.
- AUC and ROC Curve: The AUC score quantifies the overall ability of the model to distinguish between classes, while the ROC curve provides insights into performance across various thresholds.

Model Performance

The Logistic Regression model provided us with a baseline performance. While it achieved a reasonable accuracy, its F1 score, precision, and recall were low, indicating that the model struggled with identifying churn customers effectively.
The modified Logistic Regression model, after applying SMOTE and MinMax scaling, performed better in terms of recall, successfully identifying churn instances. However, it showed a lower precision, meaning there were many false positives. The overall accuracy also dropped compared to the baseline Logistic Regression model.
The Decision Tree model performed better overall, showing improved precision and recall. It successfully identified churn cases, though it was slightly less accurate than other models.
The Random Forest model exhibited the highest accuracy and F1 score among all models. It had excellent precision, meaning it correctly identified churn customers, and was also good at minimizing false positives. It balanced recall and precision well, making it a very effective model.
In addition to the model summaries, we visualized the ROC curve and AUC to more precisely compare model performance, assessing the balance between true positive and false positive rates and providing a single value to quantify overall effectiveness as seen below:

Based on the results, the Random Forest Classifier outperformed all other models. It achieved the highest accuracy and precision, and balanced recall effectively. Therefore, we recommend using this model for predicting customer churn in order to maximize prediction accuracy and minimize false positives.

## CONCLUSIONS
Through data analysis and model building, we identified key features such as 'account length', 'customer service calls', 'churn', 'international plan_yes', 'voice mail plan_yes', 'total calls', and 'total charge' that significantly predict customer churn. The Decision Tree Classifier performed well, while the Linear Regression model struggled despite SMOTE and MinMax scaling. The Random Forest Classifier excelled, achieving over 90% accuracy, demonstrating that ensemble methods like Random Forest, along with decision trees, are highly effective for churn prediction.

## RECOMMENDATIONS AND NEXT STEPS

- Enhance Feature Selection:
Further exploration of customer behavior and satisfaction related features could improve model accuracy in predicting churn and help identify key factors leading to service discontinuation.
- Optimize Model Performance:
Fine-tuning the Random Forest and Decision Tree models, along with experimenting with alternative sampling techniques, can help improve prediction accuracy, especially in identifying at-risk customers.
- Focus on Customer Retention Strategies:
Use the model's predictions to implement targeted retention efforts, such as personalized interventions or loyalty programs, and continuously monitor the model to ensure its effectiveness in predicting when customers are likely to discontinue their service.

## REFERENCES
These are the references that used:
- [Worldfolio](https://www.theworldfolio.com/company/syriatel-syria/197/)
- [Kaggle](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset/discussion/448926)
- [Productplan.com](Productplan.com)
- [Moringa School](moringaschool.com) classroom resources
