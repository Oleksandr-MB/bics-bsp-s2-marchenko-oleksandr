import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
import warnings
import sys
warnings.filterwarnings("ignore")

class LSTM(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()  # Call the constructor of the parent class, nn.Module
        self.hidden_size = hidden_size  # Set the number of neurons in each LSTM layer
        self.num_layers = num_layers  # Set the number of LSTM layers

        # Instantiate the LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Instantiate the fully connected layer
        self.add_layer = nn.Linear(hidden_size, output_size)
        # Instantiate the activation function
        self.activation = nn.ReLU()

    def forward(self, X):
        # Create initial hidden and cell states
        hidden_init = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(X.device)
        cell_init = torch.zeros(self.num_layers, X.size(0), self.hidden_size).to(X.device)

        # Forward propagate the LSTM
        out, _ = self.lstm(X, (hidden_init, cell_init))
        # Select the output features from the last time step
        out = out[:, -1, :]
        # Apply the activation function
        out = self.activation(out)
        # Apply the fully connected layer
        out = self.add_layer(out)

        return out

class Predictioner:
    def __init__(self, dataset_num, division, region, location, building_type, output_size, lookback):
        self.dataset_num = dataset_num
        self.division = division
        self.region = region
        self.location = location
        self.building_type = building_type

        self.set_file_path()
        self.set_target_column()
        self.set_some_lstm_params(lookback, output_size)
        
    
    def set_target_column(self):
        self.target_column = "Number of Sales" if self.dataset_num == "ds6" else "Median Price"
    
    def set_file_path(self):
        self.path = f"Dataset/Derived/{self.dataset_num}"

    def set_some_lstm_params(self, lookback, output_size):
        self.mm_sc = MinMaxScaler()
        self.std_sc = StandardScaler()

        self.lookback = lookback
        self.output_size = output_size

    def read_data(self): 
        
        if self.location == "":
            file = f"{self.path}/{self.division}{self.building_type}/{self.region}.csv"
        else:
            file = f"{self.path}/{self.division}{self.building_type}/{self.location}.csv"
            
        df_location = pd.read_csv(file, index_col=0, header=0)
        df_location.rename(columns={df_location.columns[0]: self.target_column}, inplace=True)
        df_location.dropna(subset=[self.target_column], inplace=True)
        
        #return df_location[[self.target_column, "GDP", "Population", "Inflation Rate", "Interest Rate"]][:-12] # Uncomment this line for assessment

        return df_location[[self.target_column, "GDP", "Population", "Inflation Rate", "Interest Rate"]]
    
    def prepare_Xy(self):
        df = self.read_data()  # Read the data
        # Split the dataframe into input features (X) and target variable (y)
        X, y = df.drop(self.target_column, axis=1), df[self.target_column].values

        return X, y  # Return the input features and target variable

    def transform_Xy(self, X, y):
        # Scale the input features (X) using StandardScaler
        X_trans = self.std_sc.fit_transform(X)

        # Scale the target variable (y) using MinMaxScaler
        y_trans = self.mm_sc.fit_transform(y.reshape(-1, 1)) 

        return X_trans, y_trans  # Return the transformed input features and target variable

    # Function to split sequences into input/output samples for the LSTM
    def split_sequences(self, in_seq, out_seq, lookback, output_size):
        X, y = [], []  # Instantiate X and y
        for i in range(len(in_seq)):
            # Find the end of the input and output sequence
            end_in_x = i + lookback
            end_out_x = end_in_x + output_size - 1

            # Check for exceeding the dataset
            if end_out_x > len(in_seq): 
                break

            # Gather input and output of the pattern
            seq_x, seq_y = in_seq[i:end_in_x], out_seq[end_in_x-1:end_out_x, -1]
            X.append(seq_x), y.append(seq_y)  # Append the sequences to X and y
        return np.array(X), np.array(y)  # Return the sequences as numpy arrays

    # Function to create tensors for LSTM input/output
    def create_tensors(self, lookback, num_pred):
        X, y = self.prepare_Xy() # Prepare input/output data
        X_trans, y_trans = self.transform_Xy(X, y) # Transform input/output data
        self.num_pred = num_pred # Number of predictions
        X_std_sc, y_mm_sc = self.split_sequences(X_trans, y_trans, lookback, num_pred) # Split sequences

        split_point = int(0.85*len(X)) # Split point for training/test
        self.split = split_point # Save the split

        # Split data into training/test sets
        X_train, y_train = X_std_sc[:split_point], y_mm_sc[:split_point] # Training
        X_test, y_test = X_std_sc[split_point:], y_mm_sc[split_point:] # Test

        # Convert data to tensors
        X_train_tensors = torch.Tensor(X_train)
        X_test_tensors = torch.Tensor(X_test)

        # Reshape tensors for LSTM
        X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], lookback, X_train_tensors.shape[2]))
        X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], lookback, X_test_tensors.shape[2])) 
        y_train_tensors_final = torch.Tensor(y_train)
        y_test_tensors_final = torch.Tensor(y_test)

        # Print shapes of final tensors
        print("Training Shape:", X_train_tensors_final.shape, y_train_tensors_final.shape)
        print("Testing Shape:", X_test_tensors_final.shape, y_test_tensors_final.shape) 

        self.X_test_tensors_final = X_test_tensors_final # Save final test tensors
        return X_train_tensors_final, X_test_tensors_final, y_train_tensors_final, y_test_tensors_final


    # Function to train the LSTM model
    def train_model(self, num_ep, lstm, optimiser, loss_fn, X_train, y_train, X_test, y_test):
        for epoch in range(num_ep): # For each epoch
            lstm.train() # Set model to training mode
            outputs = lstm.forward(X_train) # Forward pass
            lstm.eval() # Set model to evaluation mode
            optimiser.zero_grad() # Zero gradients

            # Calculate loss
            train_loss = loss_fn(outputs, y_train)

            # Backpropagation
            train_loss.backward() # Compute gradients
            optimiser.step() # Update weights

            # Evaluate on test data
            test_preds = lstm(X_test)
            test_loss = loss_fn(test_preds, y_test)

            # Print loss every 100 epochs
            if epoch % 100 == 99:     
                print("Epoch: %d, train loss: %1.5f, test loss: %1.5f" % (epoch, train_loss.item(),  test_loss.item()))

    # Function to prepare data for plotting
    def prepare_to_plot(self, df, lstm):
        # Transform data
        df_X_sc = self.std_sc.transform(df.drop(self.target_column, axis=1)) # Transform input data
        df_y_sc = self.mm_sc.transform(df[self.target_column].values.reshape(-1, 1)) # Transform output data

        # Split sequences
        df_X_sc, df_y_sc = self.split_sequences(df_X_sc, df_y_sc, self.lookback, self.output_size)

        # Convert to tensors
        df_X_sc = torch.Tensor(df_X_sc)
        df_y_sc = torch.Tensor(df_y_sc)
        df_X_sc = torch.reshape(df_X_sc, (df_X_sc.shape[0], self.lookback, df_X_sc.shape[2]))

        # Generate predictions
        df_predict = lstm(df_X_sc) # Forward pass
        df_predict = df_predict.data.numpy() # Convert to numpy array
        df_predict = self.mm_sc.inverse_transform(df_predict) # Inverse transform predictions

        # Split predictions into training/test sets
        train_predict = df_predict[:self.split]
        test_predict = df_predict[self.split:]

        return train_predict, test_predict

    
    def predict_with_LSTM(self, lstm, X_test):
        predictions_lstm = []
        input = X_test[-1].unsqueeze(0)  # Take the last sequence from the test data
        output = lstm(input).unsqueeze(0)

        # Inverse scale the predictions
        predictions_lstm = self.mm_sc.inverse_transform(output.detach().numpy().reshape(-1, 1)).flatten()
        print("LSTM Predictions:", predictions_lstm)

        return predictions_lstm

    def predict_with_LR(self, X, y, deg):
        # Create an array of indices representing time steps
        X_i = np.array([_ for _ in range(len(X))]).reshape(-1, 1)

        # Initialize and fit a simple linear regression model
        regression_simple = LinearRegression()
        regression_simple.fit(X_i, y)
        # Predict using the simple linear regression model
        predictions_lr_simple = regression_simple.predict(X_i)

        # Create polynomial features based on the degree specified
        poly_features = PolynomialFeatures(degree=deg)
        XP = poly_features.fit_transform(X_i)
        # Initialize and fit a polynomial regression model
        lr_poly = LinearRegression()
        lr_poly.fit(XP, y)
        # Predict using the polynomial regression model
        predictions_lr_poly = lr_poly.predict(XP)

        # Generate future predictions using both models
        for i in range(len(X), len(X) + self.output_size):
            # Predict the next time step with the simple linear regression model
            next_prediction_lr_simple = regression_simple.predict(np.array([i]).reshape(1, -1))
            # Append the prediction to the array of predictions
            predictions_lr_simple = np.append(predictions_lr_simple, next_prediction_lr_simple)

            # Predict the next time step with the polynomial regression model
            next_prediction_lr_poly = lr_poly.predict(poly_features.transform(np.array([i]).reshape(1, -1)))
            # Append the prediction to the array of predictions
            predictions_lr_poly = np.append(predictions_lr_poly, next_prediction_lr_poly)

        # Print the predictions
        print("Simple LR predictions", predictions_lr_simple[-self.output_size:])
        print(f"Polynomial LR (degree {deg}) predictions", predictions_lr_poly[-self.output_size:])

        # Return the predictions from both models
        return predictions_lr_simple, predictions_lr_poly

    def plot_data(self, df, lstm):
        X, y = self.prepare_Xy()
        big_dummy = np.array([[np.nan] for _ in range(len(y))])
        small_dummy = np.array([[np.nan] for _ in range(self.output_size)])

        # Plot the actual data
        plt.figure(figsize=(12,6))
        date_range = pd.date_range(start="01/01/1996", periods=len(y)+self.output_size, freq="MS")
        
        #LR
        plot_lr_simple, plot_lr_poly = self.predict_with_LR(X, y, deg=3)
        plt.plot(date_range, plot_lr_simple, label="Simple LR Predictions", color="purple")
        plt.plot(date_range, plot_lr_poly, label="Polynomial LR Predictions", color="teal")

        y = np.concatenate((y, small_dummy.flatten()))
        plt.plot(date_range, y, label="Actual Data", color="blue")

        # LSTM
        train_predict, test_predict = self.prepare_to_plot(df, lstm)

        # Plot the training predictions
        train_pred_plot = np.empty_like(y)
        train_pred_plot[:] = np.nan
        train_pred_plot[:len(train_predict)] = train_predict[:, 0]
        plt.plot(date_range, train_pred_plot, label="LSTM Training", color="orange")

        # Plot the test predictions
        test_pred_plot = np.empty_like(y)
        test_pred_plot[:] = np.nan
        test_pred_plot[len(train_predict)+self.output_size-1:-self.output_size-self.lookback+1] = test_predict[:, 0]

        plt.plot(date_range, test_pred_plot, label="LSTM Test", color="red")

        # Plot n-step predictions
        n_step_pred_plot = np.empty_like(y)
        n_step_pred_plot[:] = np.nan
        n_step_pred_plot[len(big_dummy):] = self.predict_with_LSTM(lstm, self.X_test_tensors_final)
        plt.plot(date_range, n_step_pred_plot, label="LSTM Predictions", color="green")

        plt.axvline(x=date_range[self.split-1], color="grey", linestyle="--") 
        plt.axvline(x=date_range[len(y)-self.output_size-1], color="grey", linestyle="--") 
        
        plt.xlabel("Date")
        plt.ylabel(self.target_column)
        plt.title("Time-Series Prediction")
        plt.legend()
        plt.show()

     
def main():
    dataset_num, division, region, location, building_type, window = sys.argv[1:]
    
    lookback = 3
    output_size = int(window) # number of output values 
    input_size = 4 # number of features
    hidden_size = 64 # number of features in hidden state
    num_layers = 1 # number of stacked lstm layers
    num_ep = 1000 # number of epochs
    learning_rate = 0.02 # sensitivity to outliners

    #output_size = 12 # Uncomment this line for assessment

    # Initialize the Prediction class
    try:
        # Deal with edge cases that don't have certain types of dwelling
        if location in ["City of London", "Isles of Scilly"] and building_type != "All":
            building_type = "All"

        prediction = Predictioner(dataset_num, division, region, location, building_type, output_size, lookback)
        df = prediction.read_data()
        print(df)
    except:
        prediction = Predictioner("ds6", "1", "England and Wales", "", "All", output_size, lookback)
        df = prediction.read_data()
        print(df)

    # Create tensors
    X_train_tensors_final, X_test_tensors_final, y_train_tensors_final, y_test_tensors_final = prediction.create_tensors(lookback, output_size)

    lstm = LSTM(output_size, input_size, hidden_size, num_layers) 

    loss_fn = torch.nn.MSELoss() # use MSE loss function
    optimiser = torch.optim.Adam(lstm.parameters(), lr=learning_rate) 

    # Train the model
    prediction.train_model(num_ep, lstm, optimiser, loss_fn, X_train_tensors_final, y_train_tensors_final, X_test_tensors_final, y_test_tensors_final)
    
    # Plot the data
    prediction.plot_data(df, lstm)
    
if __name__ == "__main__":
    main()