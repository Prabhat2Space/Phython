### **Software Requirements Specification (SRS) Document**

---

#### **1. Introduction**

##### **1.1 Purpose**
The purpose of this document is to provide a detailed Software Requirements Specification (SRS) for the development of the AirCast project, an LSTM-based Air Quality Index (AQI) prediction model. 
This document outlines the functional and non-functional requirements, system architecture, design considerations, and constraints.

##### **1.2 Scope**
AirCast is a predictive model designed to forecast future AQI levels using historical air pollution data. 
The project will deliver a machine learning model built using PyTorch, which can be deployed to provide accurate AQI predictions. 
The scope includes the development of the model, its integration into existing environmental monitoring systems, and the creation of a user interface for interacting with the model.

##### **1.3 Definitions, Acronyms, and Abbreviations**
- **AQI:** Air Quality Index
- **LSTM:** Long Short-Term Memory (a type of neural network)
- **SRS:** Software Requirements Specification
- **FRS:** Functional Requirements Specification
- **API:** Application Programming Interface

##### **1.4 References**
- **[PyTorch Documentation](https://pytorch.org/docs/stable/index.html)**
- **[WHO Air Quality Guidelines](https://www.who.int/air-pollution)**
- **[Sklearn Documentation](https://scikit-learn.org/stable/documentation.html)**

##### **1.5 Overview**
This document provides a comprehensive outline of the system's requirements, including functional and non-functional requirements, system features, design constraints, and user interface design. 
It serves as a guide for the development, testing, and deployment phases of the project.

---

#### **2. Overall Description**

##### **2.1 Product Perspective**
The AirCast project is a standalone system that can be integrated into existing environmental monitoring platforms.
It will utilize historical AQI data to predict future air quality, providing actionable insights for city planners, health departments, and the general public.

##### **2.2 Product Features**
- Predict AQI levels using historical data.
- Provide real-time AQI forecasts based on the model's predictions.
- Support integration with external systems via API.
- Allow for visualization of AQI trends and predictions.

##### **2.3 User Classes and Characteristics**
- **Environmental Analysts:** Use the system to monitor and predict air quality for urban planning and public health initiatives.
- **Developers:** Integrate the model with existing systems and customize the model's features.
- **General Public:** Access AQI predictions through a user-friendly interface for personal health and safety decisions.

##### **2.4 Operating Environment**
- **Platform:** The system will be developed using PyTorch and Python.
- **Deployment:** Can be deployed on cloud platforms such as AWS, GCP, or on local servers.
- **Database:** The system will use a relational or NoSQL database to store historical AQI data.

##### **2.5 Design and Implementation Constraints**
- The system must handle large datasets and perform computations efficiently.
- The model should be scalable to accommodate different regions with varying AQI patterns.
- The system must be compatible with existing environmental monitoring infrastructure.

##### **2.6 Assumptions and Dependencies**
- Historical AQI data is assumed to be available and clean.
- The system assumes access to modern computational resources for model training and deployment.
- Dependencies include PyTorch, Scikit-learn, and data preprocessing libraries.

---

#### **3. Functional Requirements Specification (FRS)**

##### **3.1 System Features**

###### **3.1.1 Data Ingestion**
**Description:** The system shall ingest historical AQI data from various sources, including CSV files and APIs.
- **Input:** Historical AQI data in CSV format.
- **Output:** Processed and scaled AQI data ready for model training.
- **Functionality:** 
  - Ingest data from files and APIs.
  - Normalize data using Min-Max Scaling.
  - Handle missing data and outliers.

###### **3.1.2 Data Preprocessing**
**Description:** The system shall preprocess the ingested data to prepare it for model training.
- **Input:** Raw AQI data.
- **Output:** Preprocessed data suitable for training.
- **Functionality:** 
  - Convert date-time data to appropriate formats.
  - Normalize and scale data.
  - Create sequences of data for LSTM training.

###### **3.1.3 Model Training**
**Description:** The system shall train an LSTM model using the preprocessed AQI data.
- **Input:** Preprocessed AQI data sequences.
- **Output:** A trained LSTM model.
- **Functionality:** 
  - Train the model using a specified number of epochs and batch size.
  - Implement early stopping to prevent overfitting.
  - Save the trained model to disk.

###### **3.1.4 AQI Prediction**
**Description:** The system shall predict future AQI levels using the trained LSTM model.
- **Input:** Sequence of recent AQI data.
- **Output:** Predicted AQI value for the next day.
- **Functionality:** 
  - Load the trained model.
  - Predict AQI for the given sequence.
  - Scale the predicted value back to its original range.

###### **3.1.5 Data Visualization**
**Description:** The system shall visualize the predicted AQI against actual AQI values.
- **Input:** Predicted AQI values and actual AQI values.
- **Output:** A plot showing the comparison between predicted and actual AQI.
- **Functionality:** 
  - Plot the actual AQI values over time.
  - Overlay the predicted AQI values.
  - Provide interactive visualizations for user analysis.

###### **3.1.6 API Integration**
**Description:** The system shall expose an API for external systems to interact with the model.
- **Input:** API requests with AQI data sequences.
- **Output:** Predicted AQI values.
- **Functionality:** 
  - Accept API requests and return predictions.
  - Ensure secure and efficient API communication.

---

#### **4. Non-Functional Requirements**

##### **4.1 Performance Requirements**
- The system shall provide AQI predictions within 1 second of receiving the input sequence.
- The system shall be capable of handling datasets with millions of records without performance degradation.

##### **4.2 Security Requirements**
- The system shall ensure data integrity by validating all inputs.
- The API shall be protected with authentication and encryption to prevent unauthorized access.

##### **4.3 Usability Requirements**
- The user interface shall be intuitive and easy to navigate for non-technical users.
- Documentation shall be provided to guide users through the system's features and API usage.

##### **4.4 Reliability Requirements**
- The system shall have an uptime of 99.9% to ensure continuous availability.
- The system shall implement error handling and logging to manage unexpected issues.

##### **4.5 Scalability Requirements**
- The system shall be scalable to support an increasing number of users and datasets.
- The architecture shall allow for easy addition of new features and models.

---

#### **5. System Architecture and Design**

##### **5.1 System Architecture**
- **Data Layer:** Handles the ingestion, storage, and preprocessing of AQI data.
- **Model Layer:** Implements the LSTM model for AQI prediction.
- **API Layer:** Exposes the prediction functionality to external systems via a RESTful API.
- **Presentation Layer:** Provides a user interface for data visualization and interaction with the model.

##### **5.2 Database Design**
- **Tables:**
  - `AQI_Data`: Stores historical AQI data with timestamps.
  - `Model_Logs`: Records the performance and parameters of trained models.
  - `Predictions`: Logs all predictions made by the model for analysis.

##### **5.3 Component Design**
- **Data Ingestion Component:** Handles data extraction from various sources.
- **Preprocessing Component:** Cleans and normalizes data for model input.
- **LSTM Model Component:** Implements the core predictive model.
- **Visualization Component:** Creates visual representations of predictions.
- **API Component:** Manages communication with external systems.

---

#### **6. User Interface Design**

##### **6.1 Overview**
The user interface will be a web-based dashboard allowing users to upload data, view predictions, and analyze trends.

##### **6.2 Main Pages**
- **Dashboard:** Displays overall system status and recent predictions.
- **Data Upload:** Allows users to upload historical AQI data.
- **Prediction Viewer:** Shows the predicted AQI alongside actual values.
- **API Documentation:** Provides details on how to interact with the system via API.

---

#### **7. Conclusion**
This document outlines the complete requirements and design for the AirCast project. 
It will serve as the foundation for the development, testing, and deployment of the system, ensuring that it meets the specified objectives and delivers accurate AQI predictions.

---

### **Functional Requirements Specification (FRS) Document**

---

#### **1. Introduction**

##### **1.1 Purpose**
The purpose of this Functional Requirements Specification (FRS) document is to detail the functional requirements for the AirCast project, an LSTM-based model for predicting Air Quality Index (AQI).
This document will guide the development team in implementing the features necessary to meet the project objectives.

---

#### **2. System Overview**

The AirCast project aims to predict future AQI levels using historical data, providing insights into air quality trends. 
The system will ingest data, preprocess it, train a predictive model, and visualize the results. It will also offer an API for integration with external systems.

---

#### **3. Functional Requirements**

##### **3.1 Data Ingestion**
- **Requirement:** The system shall allow users to upload historical AQI data in CSV format.
- **Description:** Users will upload a CSV file containing historical AQI data, which the system will ingest and store in the

 database.
- **Input:** CSV file with AQI data.
- **Output:** Data stored in the `AQI_Data` table.
- **Priority:** High

##### **3.2 Data Preprocessing**
- **Requirement:** The system shall preprocess the ingested data to ensure it is ready for model training.
- **Description:** The system will normalize, clean, and sequence the data to create inputs suitable for LSTM model training.
- **Input:** Raw AQI data from the database.
- **Output:** Preprocessed data sequences.
- **Priority:** High

##### **3.3 Model Training**
- **Requirement:** The system shall train an LSTM model using the preprocessed AQI data.
- **Description:** The system will train an LSTM model on the historical data, adjusting weights to minimize prediction errors.
- **Input:** Preprocessed data sequences.
- **Output:** Trained LSTM model saved for future predictions.
- **Priority:** High

##### **3.4 AQI Prediction**
- **Requirement:** The system shall predict AQI values using the trained model.
- **Description:** Given a sequence of recent AQI data, the system will predict the AQI for the next day.
- **Input:** Sequence of recent AQI data.
- **Output:** Predicted AQI value.
- **Priority:** High

##### **3.5 Data Visualization**
- **Requirement:** The system shall visualize the predicted AQI alongside actual AQI values.
- **Description:** The system will provide visual charts to compare predicted and actual AQI values over time.
- **Input:** Predicted AQI values and actual AQI values.
- **Output:** Visual representation of AQI trends.
- **Priority:** Medium

##### **3.6 API Integration**
- **Requirement:** The system shall expose an API for external systems to retrieve AQI predictions.
- **Description:** External systems can request AQI predictions via an API, and the system will return the predicted values.
- **Input:** API request with necessary parameters.
- **Output:** Predicted AQI value.
- **Priority:** Medium

##### **3.7 User Interface**
- **Requirement:** The system shall provide a user interface for data upload, model training, and result visualization.
- **Description:** The web-based UI will allow users to interact with the system, upload data, and view predictions.
- **Input:** User actions (e.g., data upload, viewing predictions).
- **Output:** Responsive UI displaying relevant data and predictions.
- **Priority:** Medium

---

#### **4. Use Case Scenarios**

##### **4.1 Use Case 1: Upload Historical AQI Data**
- **Actor:** Environmental Analyst
- **Scenario:** The user uploads a CSV file containing historical AQI data.
- **Steps:**
  1. The user logs into the system.
  2. The user navigates to the Data Upload page.
  3. The user selects the CSV file and clicks "Upload."
  4. The system ingests and processes the file, storing the data in the database.
  5. The user receives a confirmation message.

##### **4.2 Use Case 2: Train LSTM Model**
- **Actor:** Data Scientist
- **Scenario:** The user initiates the training of the LSTM model using preprocessed data.
- **Steps:**
  1. The user logs into the system.
  2. The user navigates to the Model Training page.
  3. The user clicks "Train Model."
  4. The system preprocesses the data and begins model training.
  5. The user receives a notification once training is complete.

##### **4.3 Use Case 3: Predict AQI**
- **Actor:** General Public
- **Scenario:** The user requests a prediction for the next day's AQI.
- **Steps:**
  1. The user accesses the prediction page.
  2. The user provides necessary input (e.g., recent AQI data).
  3. The system predicts the next day's AQI and displays it.
  4. The user views the predicted AQI.

---

#### **5. Non-Functional Requirements**

##### **5.1 Performance**
- The system shall process and predict AQI within 1 second of receiving input.
- The system shall support concurrent users without significant performance degradation.

##### **5.2 Security**
- All user data, including predictions, shall be encrypted during storage and transmission.
- The system shall implement role-based access control (RBAC) to ensure only authorized users can access certain features.

##### **5.3 Usability**
- The UI shall be responsive and accessible, ensuring usability across devices.
- The system shall include detailed tooltips and help documentation.

---

#### **6. Design Constraints**

##### **6.1 Scalability**
- The system architecture shall support the addition of new data sources and regions without significant changes to the core system.

##### **6.2 Compatibility**
- The system shall be compatible with major browsers (Chrome, Firefox, Edge) and operating systems (Windows, Linux, macOS).

---

#### **7. Conclusion**

The FRS document outlines the functional requirements for the AirCast project, serving as a detailed guide for the development team to build and deploy a reliable AQI prediction system.

---

This SRS and FRS document provides a comprehensive foundation for the AirCast project, detailing all necessary requirements and design considerations for successful development and deployment.
