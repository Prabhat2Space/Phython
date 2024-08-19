### **Software Requirements Specification (SRS)**

#### **1. Objective**
The objective of this project is to analyze the cost of living indices by country and predict the cost of living class (Low, Medium, High) using a RandomForest classification model. 
To address class imbalance, the SMOTE technique is used, and the model's hyperparameters are optimized to enhance performance. TensorFlow is employed for the implementation.

#### **2. Scope**
This project involves the following key activities:
- Data loading, cleaning, and preprocessing.
- Data analysis and visualization to understand the distribution of cost of living across countries.
- Building a RandomForest classification model to predict the cost of living class.
- Applying SMOTE to handle class imbalance.
- Optimizing the RandomForest model using hyperparameter tuning.
- Evaluating the model's performance and providing detailed analysis.
- Using TensorFlow for model development and deployment.

#### **3. Functional Requirements**
- **Data Loading and Preprocessing**: The system shall load the "Cost_of_Living_Index_by_Country_2024.csv" dataset and preprocess it by handling missing values, normalizing features, and splitting the dataset into training and testing sets.
- **Data Analysis**: The system shall sort the data by the Cost of Living Index and identify the top 10 most expensive and cheapest countries. Visualizations will be created to compare these countries across various indices.
- **Model Development**: The system shall develop a RandomForest classifier using TensorFlow and integrate it with an imbalanced data handling technique (SMOTE).
- **Hyperparameter Optimization**: The system shall optimize the RandomForest model using GridSearchCV or other relevant techniques.
- **Model Evaluation**: The system shall evaluate the model's performance using accuracy, classification reports, and cross-validation scores.
- **Visualization**: The system shall visualize the results, including the top 10 most and least expensive countries, and the distribution of various indices.

#### **4. Non-Functional Requirements**
- **Performance**: The model should achieve a high level of accuracy, precision, and recall in predicting the cost of living class.
- **Scalability**: The system should be able to handle datasets of varying sizes efficiently.
- **Usability**: The system should be easy to use, with clear instructions and well-documented code.

#### **5. Assumptions**
- The dataset is assumed to be up-to-date and accurate.
- SMOTE will effectively address class imbalance issues.

#### **6. Constraints**
- The project is constrained by the availability of relevant data.
- The performance of the model may be limited by the quality and quantity of the data.

#### **7. Risks**
- Inadequate data quality may lead to poor model performance.
- Overfitting during hyperparameter tuning may affect the model's generalizability.

#### **8. Dependencies**
- Python libraries such as Pandas, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn, and TensorFlow.
- Availability of computing resources for model training and evaluation.

#### **9. Conclusion**
This SRS document outlines the requirements for developing a system to predict the cost of living class by country using a RandomForest classifier. 
The system will handle class imbalance using SMOTE and optimize model performance through hyperparameter tuning, with the implementation carried out in TensorFlow.

---

### **Functional Requirements Specification (FRS)**

#### **1. Data Loading and Preprocessing**
- **FRS1.1**: The system shall load the "Cost_of_Living_Index_by_Country_2024.csv" file.
- **FRS1.2**: The system shall handle any missing values in the dataset.
- **FRS1.3**: The system shall normalize the features to ensure consistent scaling.
- **FRS1.4**: The system shall split the dataset into training and testing sets with a default ratio of 80:20.

#### **2. Data Analysis**
- **FRS2.1**: The system shall sort the dataset by the Cost of Living Index in descending order.
- **FRS2.2**: The system shall identify and display the top 10 most expensive and cheapest countries.
- **FRS2.3**: The system shall create bar plots to visualize the cost of living and other indices for the top 10 countries.

#### **3. Model Development**
- **FRS3.1**: The system shall develop a RandomForest classification model using TensorFlow.
- **FRS3.2**: The system shall integrate SMOTE to handle class imbalance in the dataset.
- **FRS3.3**: The system shall provide functionality for hyperparameter tuning using GridSearchCV.

#### **4. Model Evaluation**
- **FRS4.1**: The system shall evaluate the model using accuracy, precision, recall, and F1-score.
- **FRS4.2**: The system shall generate a classification report for the model's performance on the test set.
- **FRS4.3**: The system shall perform cross-validation with StratifiedKFold to validate the model.

#### **5. Visualization**
- **FRS5.1**: The system shall visualize the top 10 most and least expensive countries based on the Cost of Living Index.
- **FRS5.2**: The system shall visualize the distribution of categories such as Groceries Index, Restaurant Price Index, and Rent Index.

#### **6. Documentation**
- **FRS6.1**: The system shall provide documentation for all code, including explanations for key functions and processes.
- **FRS6.2**: The system shall include comments within the code to ensure maintainability.

#### **7. User Interface**
- **FRS7.1**: The system shall provide a command-line interface for executing the analysis and predictions.

#### **8. Conclusion**
This FRS document details the functional requirements for the system, ensuring it meets the project's objectives through comprehensive data analysis, model development, evaluation, and visualization.

---

### **Design Document**

#### **1. Introduction**
This design document provides a detailed architecture and design for the system that predicts the cost of living class using a RandomForest classification model and TensorFlow, with SMOTE for class imbalance handling.

#### **2. System Architecture**
- **Data Layer**: Responsible for loading, cleaning, and preprocessing the data from the "Cost_of_Living_Index_by_Country_2024.csv" file.
- **Analysis Layer**: Performs sorting, analyzing, and visualizing the data to provide insights into the cost of living across countries.
- **Modeling Layer**: Implements the RandomForest classifier using TensorFlow, integrates SMOTE, and performs hyperparameter tuning.
- **Evaluation Layer**: Evaluates the model using various metrics and visualizes the performance.
- **User Interface Layer**: Provides a command-line interface for interaction with the system.

#### **3. Data Flow**
1. **Data Loading**: The CSV file is loaded into a Pandas DataFrame.
2. **Data Preprocessing**: Missing values are handled, features are normalized, and the data is split into training and testing sets.
3. **Data Analysis**: The data is sorted, and the top 10 most and least expensive countries are identified. Bar plots are created to visualize the indices.
4. **Model Development**: A RandomForest model is trained using the processed data. SMOTE is applied to handle class imbalance.
5. **Hyperparameter Tuning**: GridSearchCV is used to optimize the model's parameters.
6. **Model Evaluation**: The model's performance is evaluated, and a classification report is generated.
7. **Visualization**: The results are visualized, including the top 10 countries and the distribution of various indices.

#### **4. Algorithms and Techniques**
- **RandomForest Classifier**: Used for predicting the cost of living class.
- **SMOTE**: Applied to generate synthetic samples for the minority class to address class imbalance.
- **GridSearchCV**: Employed for hyperparameter optimization.
- **StratifiedKFold Cross-Validation**: Used to validate the model during training.

#### **5. Implementation Details**
- **Language**: Python
- **Libraries**: Pandas, Matplotlib, Seaborn, Scikit-learn, Imbalanced-learn, TensorFlow
- **Pipeline**: The system uses an imPipeline to integrate SMOTE with the RandomForest model.

#### **6. User Interface Design**
- **Command-Line Interface**: Users interact with the system via the command line, executing scripts to perform analysis, train the model, and view results.

#### **7. Testing**
- **Unit Testing**: Tests will be written to validate individual functions such as data loading, preprocessing, model training, and evaluation.
- **Integration Testing**: Ensures that the data flow between different layers (data, analysis, modeling, evaluation) works seamlessly.
- **Performance Testing**: The model's performance will be tested on various datasets to ensure scalability.

#### **8. Conclusion**
This design document outlines the system's architecture and design, ensuring a structured approach to developing the cost of living prediction model. 
The design integrates advanced techniques like SMOTE and TensorFlow to meet the project's objectives effectively.

---

These documents provide a comprehensive blueprint for developing and implementing the project. Let me know if you need further details or adjustments!
