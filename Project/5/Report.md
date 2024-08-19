### **Project Documentation**

---

#### **1. Project Overview**

**Project Title**: **Analysis of Cost of Living by Country Using RandomForest and SMOTE**

**Objective**:  
The primary objective of this project is to analyze cost of living indices across various countries and use machine learning techniques to predict the cost of living class (Low, Medium, High). 
The project employs a RandomForest classification model and uses SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalance. 
The model’s performance is further optimized through hyperparameter tuning, and the implementation is carried out using TensorFlow.

---

#### **2. Introduction**

**Background**:  
Cost of living is a crucial metric that influences individuals, governments, and organizations in decision-making processes related to relocation, investment, and policy formulation.
With global variations in living costs, it is important to have predictive tools that can classify and estimate the cost of living in different regions.

**Problem Statement**:  
Given a dataset with cost of living indices and other related metrics for various countries, the challenge is to predict the cost of living class (Low, Medium, High) accurately. 
The data is imbalanced, which necessitates the use of techniques like SMOTE to ensure fair model performance across all classes.

**Project Goals**:  
- To analyze the distribution of cost of living indices across countries.
- To develop a machine learning model that can predict the cost of living class.
- To address class imbalance using SMOTE.
- To optimize the model for better accuracy and performance.

---

#### **3. Methodology**

**3.1. Data Collection**:  
The dataset titled "Cost_of_Living_Index_by_Country_2024.csv" contains various indices for different countries, such as Cost of Living Index, Rent Index, Groceries Index, Restaurant Price Index, and Local Purchasing Power Index.

**3.2. Data Preprocessing**:  
- **Loading Data**: The data was loaded using Pandas and inspected for any missing values.
- **Data Cleaning**: Any missing or erroneous data points were handled appropriately.
- **Feature Scaling**: Normalization was applied to the features to ensure consistent scaling across the dataset.
- **Data Splitting**: The data was split into training and testing sets with a ratio of 80:20.

**3.3. Exploratory Data Analysis (EDA)**:  
- **Top 10 Expensive and Cheapest Countries**: The dataset was sorted by the Cost of Living Index, and the top 10 most expensive and cheapest countries were identified.
- **Visualization**: Bar plots were generated to visualize the cost of living indices for these countries, along with other indices like Groceries, Rent, and Restaurant Price Indices.

**3.4. Model Development**:  
- **Model Selection**: A RandomForest classifier was chosen for its robustness and ability to handle complex data.
- **Handling Class Imbalance**: SMOTE was used to generate synthetic samples for the minority classes, ensuring a balanced dataset for model training.
- **Hyperparameter Tuning**: GridSearchCV was employed to find the optimal hyperparameters for the RandomForest model, improving its accuracy and generalization capabilities.

**3.5. Model Evaluation**:  
- **Accuracy and Classification Report**: The model’s performance was evaluated using accuracy, precision, recall, and F1-score metrics.
  A detailed classification report was generated to provide insights into the model's performance across different classes.
- **Cross-Validation**: StratifiedKFold cross-validation was applied to ensure the model’s stability and reliability.

---

#### **4. Results and Discussion**

**4.1. Data Analysis**:  
The analysis revealed significant disparities in the cost of living across different countries. 
Countries like Switzerland, Bahamas, and Iceland were among the most expensive, while countries like Pakistan, Libya, and Egypt were among the cheapest.

**4.2. Model Performance**:  
- The RandomForest classifier, after hyperparameter tuning, achieved an accuracy of approximately 85%, with balanced performance across all classes due to the application of SMOTE.
- The classification report indicated that the model performed well in predicting the 'Medium' and 'High' cost of living classes, with slight challenges in the 'Low' class, which was initially underrepresented in the dataset.

**4.3. Visualization**:  
Visualizations provided a clear picture of how different countries compare in terms of cost of living and other related indices. These visual tools are invaluable for stakeholders in understanding global living cost dynamics.

---

#### **5. Conclusion**

**Summary**:  
This project successfully developed a machine learning model to predict the cost of living class for various countries using RandomForest and SMOTE. 
The project demonstrated the importance of handling class imbalance and optimizing model parameters to achieve accurate and reliable predictions.

**Key Learnings**:  
- The application of SMOTE is effective in addressing class imbalance, leading to better model performance.
- RandomForest is a robust classifier that can handle diverse datasets with multiple features.
- Hyperparameter tuning is crucial for improving model accuracy and generalization.

---

#### **6. Future Scope**

**6.1. Model Improvement**:  
- **Incorporate Additional Features**: Future iterations could include more economic indicators, such as GDP, unemployment rate, and inflation, to improve the model’s predictive power.
- **Experiment with Advanced Models**: Exploring other machine learning models such as Gradient Boosting Machines (GBM), XGBoost, or deep learning models could yield better performance.
- **Time Series Analysis**: Integrating time-series data to track changes in cost of living over time and predict future trends.

**6.2. Deployment**:  
- **Web Application**: Developing a web application where users can input a country's name and receive predictions and insights about its cost of living.
- **APIs**: Creating APIs that allow other systems to integrate and utilize the model for various applications, such as relocation services or economic analysis tools.

**6.3. Broader Applications**:  
- **Policy Formulation**: Governments and policymakers can use this model to understand the cost of living dynamics and make informed decisions.
- **Relocation Advisory**: The model can be adapted to provide personalized advice for individuals or businesses looking to relocate to different countries.

---


This project document encapsulates the entire workflow, from data analysis to model development, evaluation, and future prospects, providing a comprehensive overview for stakeholders and future developers.
