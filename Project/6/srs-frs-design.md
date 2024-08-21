### Software Requirements Specification (SRS)

#### 1. Introduction
**1.1 Purpose**
The purpose of this document is to outline the software requirements for the project focused on analyzing the efficiency of a High-Pressure Compressor (HPC) in turbofan engines using data analysis and machine learning techniques.
The aim is to predict HPC efficiency and understand the correlation between various engine parameters.

**1.2 Scope**
This project is intended to be a predictive maintenance tool that will help aerospace engineers monitor and maintain turbofan engines more effectively. 
The software will analyze historical data from engine performance parameters and predict potential efficiency drops in the HPC, allowing for proactive maintenance decisions.

**1.3 Definitions, Acronyms, and Abbreviations**
- **HPC:** High-Pressure Compressor
- **LPC:** Low-Pressure Compressor
- **LPT:** Low-Pressure Turbine
- **RPM:** Revolutions Per Minute
- **SRS:** Software Requirements Specification
- **FRS:** Functional Requirements Specification
- **ML:** Machine Learning

**1.4 References**
- Turbofan engine efficiency studies.
- Maintenance and Overhaul manuals for turbofan engines.
- Python and Machine Learning libraries.

**1.5 Overview**
This document outlines the functional and non-functional requirements for the software. It includes the overall description of the software, specific requirements, and external interface requirements.

#### 2. Overall Description
**2.1 Product Perspective**
The software is an independent system designed to integrate with existing engine monitoring systems. It will process engine performance data, perform correlation analysis, and predict efficiency metrics for the HPC.

**2.2 Product Functions**
- Data import and preprocessing.
- Correlation analysis of engine parameters.
- Predictive modeling for HPC efficiency.
- Visualization of results.
- Generation of maintenance recommendations.

**2.3 User Characteristics**
The primary users of this system will be aerospace engineers, maintenance personnel, and data scientists.

**2.4 Constraints**
- The system should handle large datasets efficiently.
- The system must operate within the safety and security guidelines of the aerospace industry.

**2.5 Assumptions and Dependencies**
- The system assumes the availability of historical engine performance data.
- The system depends on Python libraries such as pandas, numpy, and scikit-learn.

#### 3. Specific Requirements
**3.1 Functional Requirements**
- The system shall allow users to import engine performance datasets.
- The system shall preprocess the data to handle missing values and outliers.
- The system shall calculate correlation metrics between engine parameters.
- The system shall build and train predictive models for HPC efficiency.
- The system shall display visualizations of data correlations and predictions.
- The system shall generate reports with maintenance recommendations.

**3.2 Non-Functional Requirements**
- **Performance:** The system should provide predictions within 2 minutes of data input.
- **Usability:** The user interface should be intuitive and easy to navigate.
- **Reliability:** The system should have an uptime of 99.5%.
- **Security:** The system should ensure data integrity and confidentiality.

### Functional Requirements Specification (FRS)

#### 1. Introduction
The FRS describes the detailed functional behavior of the software. It includes all the features that the software will offer and the conditions under which each feature will operate.

#### 2. Functional Requirements
**2.1 Data Import and Preprocessing**
- The software shall allow the user to import data in CSV format.
- The software shall clean the data by handling missing values and removing outliers.

**2.2 Correlation Analysis**
- The software shall calculate the Pearson correlation coefficient for each pair of engine parameters.
- The software shall display a heatmap of the correlation matrix.

**2.3 Predictive Modeling**
- The software shall offer multiple regression models (e.g., Linear Regression, Random Forest) for predicting HPC efficiency.
- The software shall allow users to choose the best model based on evaluation metrics such as R-squared and Mean Absolute Error (MAE).

**2.4 Visualization**
- The software shall generate plots for data visualization, including scatter plots and pair plots.
- The software shall provide a dashboard to visualize model predictions and efficiency trends.

**2.5 Reporting**
- The software shall generate a PDF report summarizing the analysis results and maintenance recommendations.

### Design Document

#### 1. Introduction
The design document describes the architecture of the software, including the data flow, system components, and their interactions.

#### 2. System Architecture

**2.1 Block Diagram**

```plaintext
+----------------+       +--------------------+       +-----------------------+
|   Data Import  | ----> | Data Preprocessing | ----> |  Correlation Analysis  |
+----------------+       +--------------------+       +-----------------------+
         |                           |                            |
         V                           V                            V
+-----------------+        +--------------------+       +-----------------------+
| Predictive Model| <----> |   Visualization    | <----> |   Report Generation   |
+-----------------+        +--------------------+       +-----------------------+
```

**2.2 Data Flow Diagram**

**Level 0: Context Diagram**

```plaintext
      +----------------------------+
      |       User Interface        |
      +----------------------------+
                  |
                  V
        +---------------------+
        |  Data Import Module  |
        +---------------------+
                  |
                  V
        +---------------------+
        | Data Processing Unit|
        +---------------------+
                  |
                  V
        +---------------------+
        |   Prediction Model  |
        +---------------------+
                  |
                  V
        +---------------------+
        |   Visualization     |
        +---------------------+
                  |
                  V
        +---------------------+
        |  Report Generation  |
        +---------------------+
```

#### 3. Detailed Design

**3.1 Data Import Module**
- This module reads data from a CSV file.
- It ensures that data types are consistent and that missing values are handled.

**3.2 Data Preprocessing Unit**
- This unit handles data cleaning, normalization, and feature selection.
- It outputs a clean dataset ready for analysis.

**3.3 Correlation Analysis**
- This module computes the correlation matrix and generates a heatmap.
- It identifies the strongest correlations for further analysis.

**3.4 Predictive Model**
- Multiple regression models are implemented in this module.
- The model selection is based on performance metrics.

**3.5 Visualization**
- This module creates visual representations of data and predictions.
- It provides an interactive dashboard for users to explore the results.

**3.6 Report Generation**
- This module compiles the results into a PDF report.
- It includes graphs, tables, and maintenance recommendations.

#### 4. Flowchart

**Data Processing and Model Training Flowchart**

```plaintext
  Start
    |
    V
[Load Dataset] --> [Clean Data] --> [Feature Selection] --> [Train Model] --> [Evaluate Model]
    |                                                                                   |
    V                                                                                   V
[Visualize Data] <------------------------------------------------------------------- [Generate Report]
    |
    V
  End
```

This flowchart summarizes the sequence of steps involved in processing the data, training the model, evaluating the results, and generating a report. 
Each step feeds into the next, ensuring that the system operates in a logical and efficient manner.
