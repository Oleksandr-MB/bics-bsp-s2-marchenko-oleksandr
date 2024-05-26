import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import autocorrelation_plot
from scipy import signal
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import lag_plot
import sys

class Data_Analizer:
    def __init__(self, dataset_num, division_by, region, location):
        self.dataset_num = dataset_num
        self.division = division_by
        self.region = region
        self.location = location

        self.types_d = {"All": "a", "Detached": "b", "Semi-detached": "c", "Terraced": "d", "Flats and Maisonets": "e"}
        self.type_codes = list(self.types_d.values())
        self.labels = list(self.types_d.keys())
        
        self.path = f"Dataset/Derived/{self.dataset_num}"

        self.target_column = "Number of Sales" if self.dataset_num == "ds6" else "Median Price"
        
        self.plots = ["Original", "Trend", "Seasonal Component", "Stationary", "Detrended", "Deseasonalized", "Autocorrelation", "Lag"]
        self.colours = ["#003f5c", "#58508d", "#bc5090", "#ff6361", "#ffa600"]

    def read_spreadsheet(self, division, building_type, region, location): 
        if location == "":
            file = f"{self.path}/{division}{building_type}/{region}.csv"
        else:
            file = f"{self.path}/{division}{building_type}/{location}.csv"
            
        df_location = pd.read_csv(file, index_col=0, header=0)
        df_location.rename(columns={df_location.columns[0]: self.target_column}, inplace=True)
        df_location.dropna(subset=[self.target_column], inplace=True)

        return df_location[[self.target_column]]

    def decompose(self, df_location):
        multiplicative_decomposition = seasonal_decompose(df_location[self.target_column], model="multiplicative", period=12)
        return multiplicative_decomposition

    def trends(self, decomposition):
        return decomposition.trend

    def seasonality(self, decomposition):
        return decomposition.seasonal
    
    def is_stationary(self, series):
        statistic_test, p_value = adfuller(series)[:2]
        return statistic_test < 0 and p_value < 0.005
    
    def differenciate(self, by_location):
        if by_location is None:
            return None
        differentiated = by_location[self.target_column]

        while not self.is_stationary(differentiated):            
            differentiated = differentiated.diff()
            differentiated.dropna(inplace=True)

        return differentiated

    def detrend(self, series):
        detrended = pd.Series(signal.detrend(series.values).flatten(), index=series.index)
        detrended.dropna(inplace=True)
        return detrended

    def deseasonify(self, series):
        series = series.rolling(window=12).mean()
        series.dropna(inplace=True)
        return series

    def plot_data(self):
        pass
        
class Data_Analizer_1(Data_Analizer):
    def plot_data(self):
        for plot_type in self.plots:
            problematic = []
            fig, ax = plt.subplots(figsize=(8, 6))
            for i, letter in enumerate(self.type_codes):
                df_location = self.read_spreadsheet(self.division, letter, self.region, self.location)

                if df_location is None:
                    return
                    
                try:
                    decomposition = self.decompose(df_location.copy())
                except:
                    problematic.append(self.labels[i])
                    continue

                trends = self.trends(decomposition)
                seasonality = self.seasonality(decomposition)
                differentiated = self.differenciate(df_location.copy())
                detrended = self.detrend(df_location.copy())
                deseasonalized = self.deseasonify(df_location.copy())
                lags = df_location.copy()
                
                # Return the required plot
                match plot_type:
                    case "Original":
                        data = df_location[self.target_column]
                            
                    case "Trend":
                        data = trends
                            
                    case "Seasonal Component":
                        data = seasonality
                            
                    case "Stationary":
                        data = differentiated
                            
                    case "Detrended":
                        data = detrended
                            
                    case "Deseasonalized":
                        data = deseasonalized
                        
                    case "Autocorrelation":
                        autocorrelation_plot(deseasonalized, ax=ax, color=self.colours[i])
                        ax.plot([], [], color=self.colours[i], label=self.labels[i])
                        continue
                    
                    case "Lag":
                        lag_plot(lags, ax=ax, lag=3, c=self.colours[i]) 
                        ax.plot([], [], color=self.colours[i], label=self.labels[i])
                        continue
   
                data.plot(ax=ax, color=self.colours[i], label=self.labels[i])
                
            plt.title(self.location + self.region, fontsize=10)
            plt.suptitle(plot_type, fontsize=10)
            plt.tight_layout()

            ax.legend()
            plt.show()

        # Print the skipped housing types
        if len(problematic) > 0:
            print(", ".join(problematic) + " are not available in this area")
        
class Data_Analizer_2(Data_Analizer):   
    def plot_data(self):
        for plot_type in self.plots:
            problematic_loc = []
            problematic_reg = []
            
            fig, axs = plt.subplots(1, 2, figsize=(16, 6))
            for i, letter in enumerate(self.type_codes):
                df_location = self.read_spreadsheet(self.division, letter, self.region, self.location)
                df_region = self.read_spreadsheet(1, letter, self.region, "")

                if df_location is None or df_region is None:
                    return None

                try:
                    decomposition_loc = self.decompose(df_location.copy())
                except:
                    problematic_loc.append(self.labels[i])
                    continue

                try:
                    decomposition_reg = self.decompose(df_region.copy())
                except:
                    problematic_reg.append(self.labels[i])
                    continue

                trends_loc = self.trends(decomposition_loc)
                trends_reg = self.trends(decomposition_reg)

                seasonality_loc = self.seasonality(decomposition_loc)
                seasonality_reg = self.seasonality(decomposition_reg)

                differentiated_loc = self.differenciate(df_location.copy())
                differentiated_reg = self.differenciate(df_region.copy())

                detrended_loc = self.detrend(df_location.copy())
                detrended_reg = self.detrend(df_region.copy())
                
                deseasonalized_loc  = self.deseasonify(df_location.copy())
                deseasonalized_reg  = self.deseasonify(df_region.copy())

                match plot_type:
                    case "Original":
                        data_loc = df_location[self.target_column]
                        data_reg = df_region[self.target_column]
                            
                    case "Trend":
                        data_loc = trends_loc
                        data_reg = trends_reg
                            
                    case "Seasonal Component":
                        data_loc = seasonality_loc
                        data_reg = seasonality_reg
                            
                    case "Stationary":
                        data_loc = differentiated_loc
                        data_reg = differentiated_reg
                            
                    case "Detrended":
                        data_loc = detrended_loc
                        data_reg = detrended_reg
                            
                    case "Deseasonalized":
                        data_loc = deseasonalized_loc
                        data_reg = deseasonalized_reg
                        
                    case "Autocorrelation":
                        autocorrelation_plot(deseasonalized_loc, ax=axs[0], color=self.colours[i])
                        axs[0].plot([], [], color=self.colours[i], label=self.labels[i])

                        autocorrelation_plot(deseasonalized_reg, ax=axs[1], color=self.colours[i])
                        axs[1].plot([], [], color=self.colours[i], label=self.labels[i])
                        continue

                    case "Lag":
                        lag_plot(deseasonalized_loc, ax=axs[0], lag=3, c=self.colours[i])
                        axs[0].plot([], [], color=self.colours[i], label=self.labels[i])

                        lag_plot(deseasonalized_reg, ax=axs[1], lag=3, c=self.colours[i])
                        axs[1].plot([], [], color=self.colours[i], label=self.labels[i])
                        continue


                data_loc.plot(ax=axs[0], color=self.colours[i], label=self.labels[i])
                axs[0].set_title(self.location + ", " + self.region)

                data_reg.plot(ax=axs[1], color=self.colours[i], label=self.labels[i])
                axs[1].set_title(self.region)
            
            plt.suptitle(plot_type, fontsize=10)
            plt.tight_layout()

            axs[0].legend()
            
            plt.show()

        if len(problematic_loc + problematic_reg) > 0:
            print(", ".join(problematic_reg) + " are not available in this region")
            print(", ".join(problematic_loc) + " are not available in this location")

def main():
    try:
        dataset, division, region, location = sys.argv[1:5]
        if not location:
            Analizer = Data_Analizer_1(dataset, division, region, location)
        else:
            Analizer = Data_Analizer_2(dataset, division, region, location) 
    except:
        Analizer = Data_Analizer_1("ds6", "Region", "England and Wales", "") 

    Analizer.plot_data()

if __name__ == "__main__":
    main()