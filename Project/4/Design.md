### ** Design Document**

---

#### **1. Introduction**

The purpose of this Detailed Design Document is to provide an in-depth explanation of the architectural and component-level design of the AirCast project, an LSTM-based Air Quality Index (AQI) prediction system.
This document will include system block diagrams, component descriptions, data flow diagrams, and references to the code that implements these components.

---

#### **2. System Overview**

The AirCast system is designed to predict future AQI levels using historical air pollution data. 
The system consists of several key components, including data ingestion, preprocessing, model training, AQI prediction, data visualization, and API integration. 
The following sections will detail the design and implementation of each component.

---

#### **3. System Architecture**

##### **3.1 Block Diagram**

The block diagram below illustrates the high-level architecture of the AirCast system:

```
+------------------+       +------------------+       +------------------+       +------------------+       +------------------+       +------------------+
|  Data Ingestion  |  ---> | Data Preprocessing|  ---> | Model Training   |  ---> |  AQI Prediction  | --->| Data Visualization | --->|   API Integration | 
+------------------+       +------------------+       +------------------+       +------------------+       +------------------+       +------------------+
                                                                                   
```

---

#### **4. Component Design**

Each component of the system is broken down into detailed design sections, with explanations and code references where applicable.

##### **4.1 Data Ingestion Component**

**Design Overview:**
The Data Ingestion component is responsible for collecting historical AQI data from various sources such as CSV files, APIs, or databases. The data is then stored in a database for further processing.

**Key Functions:**
- **Data Ingestion:** Reads AQI data from CSV files and stores it in the database.
- **Data Validation:** Ensures that the ingested data meets the required format and contains valid values.

**Data Ingestion Code Reference:**

```python
import pandas as pd
from sqlalchemy import create_engine

def ingest_data(file_path, db_url):
    # Read the CSV file
    data = pd.read_csv(file_path)
    
    # Validate the data (this can be extended as per requirements)
    if 'AQI' not in data.columns:
        raise ValueError("Invalid data format: Missing 'AQI' column")
    
    # Store the data in the database
    engine = create_engine(db_url)
    data.to_sql('AQI_Data', con=engine, if_exists='append', index=False)

# Example usage
ingest_data('data/aqi_data.csv', 'sqlite:///aircast.db')
```

---

##### **4.2 Data Preprocessing Component**

**Design Overview:**
The Data Preprocessing component prepares the ingested data for model training. It handles tasks such as data normalization, missing value handling, and sequence creation for LSTM model training.

**Key Functions:**
- **Data Normalization:** Scales the AQI data to a range suitable for model input.
- **Sequence Creation:** Converts the time-series data into sequences required for LSTM model training.

**Data Preprocessing Code Reference:**

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_data(df):
    # Normalize the AQI data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['AQI'] = scaler.fit_transform(df[['AQI']])

    # Create sequences of data for LSTM input
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            sequences.append(seq)
        return np.array(sequences)
    
    seq_length = 30  # Example sequence length
    sequences = create_sequences(df['AQI'].values, seq_length)
    
    return sequences, scaler

# Example usage
df = pd.read_sql('SELECT * FROM AQI_Data', con=engine)
sequences, scaler = preprocess_data(df)
```

---

##### **4.3 Model Training Component**

**Design Overview:**
The Model Training component is responsible for training the LSTM model on the preprocessed AQI data. The trained model is saved for later use in prediction.

**Key Functions:**
- **Model Definition:** Defines the architecture of the LSTM model.
- **Model Training:** Trains the model using the preprocessed data.
- **Model Saving:** Saves the trained model to disk for future use.

**Model Training Code Reference:**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

def train_model(sequences, model, criterion, optimizer, epochs):
    train_loader = DataLoader(TensorDataset(torch.tensor(sequences, dtype=torch.float32)),
                              batch_size=32, shuffle=True)
    
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
    
    # Save the model
    torch.save(model.state_dict(), 'models/aqi_lstm.pth')

# Example usage
input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(sequences, model, criterion, optimizer, epochs=50)
```

---

##### **4.4 AQI Prediction Component**

**Design Overview:**
The AQI Prediction component uses the trained LSTM model to predict future AQI levels based on recent sequences of AQI data.

**Key Functions:**
- **Load Model:** Loads the trained model from disk.
- **Predict AQI:** Uses the model to predict AQI values for the next day.

**AQI Prediction Code Reference:**

```python
def predict_aqi(sequence, model, scaler):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        sequence = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        prediction = model(sequence)
        prediction = prediction.item()
    
    # Inverse scale the prediction
    prediction = scaler.inverse_transform([[prediction]])
    
    return prediction[0][0]

# Example usage
model = LSTMModel(input_size, hidden_size, num_layers, output_size)
model.load_state_dict(torch.load('models/aqi_lstm.pth'))

recent_sequence = sequences[-1]  # Use the last sequence for prediction
predicted_aqi = predict_aqi(recent_sequence, model, scaler)
print(f"Predicted AQI for tomorrow: {predicted_aqi}")
```

---

##### **4.5 Data Visualization Component**

**Design Overview:**
The Data Visualization component provides visual representations of the predicted AQI values alongside actual historical data. This helps users understand the trends and accuracy of the predictions.

**Key Functions:**
- **Plot AQI Trends:** Visualize the predicted and actual AQI values over time.

**Data Visualization Code Reference:**

```python
import matplotlib.pyplot as plt

def plot_aqi_trends(actual, predicted, title="AQI Trends"):
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label="Actual AQI", color='blue')
    plt.plot(predicted, label="Predicted AQI", color='red', linestyle='--')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("AQI")
    plt.legend()
    plt.show()

# Example usage
actual_aqi = df['AQI'].values[-len(predicted_aqi):]
plot_aqi_trends(actual_aqi, predicted_aqi)
```

---

##### **4.6 API Integration Component**

**Design Overview:**
The API Integration component provides an interface for external systems to interact with the AQI prediction model. It supports prediction requests and returns results in a standard format.

**Key Functions:**
- **API Endpoint for Prediction:** Receives AQI data and returns the predicted AQI.

**API Integration Code Reference:**

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    recent_sequence = preprocess_data(data)
    predicted_aqi = predict_aqi(recent_sequence, model, scaler)
    
    return jsonify({'predicted_aqi': predicted_aqi})

if __name__ == '__main__':
    app.run(debug=True)

# Example usage (POST request to /predict with JSON data)
# curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"data": [50, 55, 60, ...]}'
```

---

#### **5. Data Flow Diagram**

The Data Flow Diagram (DFD) illustrates the flow of data between

 the components of the system:

```plaintext
                          +------------------+
                          |    User          |
                          +------------------+
                                   |
                                   v
                          +------------------+
                          |    Data Ingestion|
                          +------------------+
                                   |
                                   v
                          +------------------+
                          | Data Preprocessing|
                          +------------------+
                                   |
                                   v
                          +------------------+
                          |  Model Training  |
                          +------------------+
                                   |
                                   v
                          +------------------+
                          |  AQI Prediction  |
                          +------------------+
                                   |
                                   v
                          +------------------+
                          |  API Integration |
                          +------------------+
                                   |
                                   v
                          +------------------+
                          | Data Visualization|
                          +------------------+
```

---

#### **6. Conclusion**

This detailed design document provides a comprehensive breakdown of the AirCast project, including the architecture, component design, data flow, and code references. 
This document serves as a guide for developers to implement, test, and deploy the system effectively.
