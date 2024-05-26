# Import necessary libraries
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import sys
import copy

class Map():
    # Initialize the class with building_type and text_file
    def __init__(self, building_type):
        self.building_type = building_type
        self.text_file = "Dataset/Derived/Text/LocalAuthorities.txt"
    
    # Function to open the map file
    def open_map(self):
        gdf = gpd.read_file("Dataset/map.geojson")
        return gdf
    
    # Function to read data from a csv file and calculate the ratio of last and previous statistics
    def read_data(self, location, dataset_num):
        df = pd.read_csv(f"Dataset/Derived/{dataset_num}/2{self.building_type}/{location}.csv", index_col=0, header=0)
        
        last = df.iloc[-1]["Statistic"]
        prev = df.iloc[-37]["Statistic"]

        if prev == 0 or last == 0:
            return 0
        
        ratio = ((last - prev) / prev)

        return ratio

    # Function to clamp the values in a list within 3 standard deviations
    def clamp(self, list):
        avg = np.average(list)
        sd = np.std(list)

        for i in range(len(list)):
            if list[i] > avg + 3*sd: list[i] = avg + 3*sd
            elif list[i] < avg - 3*sd: list[i] = avg - 3*sd
                
        return list
    
    # Function to normalize the values in a list
    def normalize(self, list):
        return [(val - min(list)) / (max(list) - min(list)) for val in list]

    # Function to plot the map
    def plot_map(self, map):
        # Copy the map for sales and prices
        map_sales, map_prices = map, copy.copy(map)

        # Initialize dictionaries to store ratios
        ratio_dict_sales = {}
        ratio_dict_prices = {}

        # Initialize sets to store locations
        locations1 = set()
        locations2 = set()

        # Read locations from the text file
        with open(self.text_file, "r") as f:
            for location in f.readlines():
                locations2.add(location.split(":")[1].strip())

        # Iterate over the rows in the map
        for index, row in map_sales.iterrows():
            location = row["lad11nm"]
            locations1.add(location)
            try:
                # Calculate ratios for sales and prices
                ratio_sales = self.read_data(location, "ds6")
                ratio_prices = self.read_data(location, "ds9")
                ratio_dict_sales[location] = ratio_sales
                ratio_dict_prices[location] = ratio_prices
            except:
                continue
        
        # Print locations present in the map but absent on the csv and vice versa
        print("Present in the map but absent on the csv (added in between 1996 - 2011):\n", locations1.difference(locations2))
        print("Present in the csv but absent on the map (abolished in between 1996 - 2011):\n", locations2.difference(locations1))

        # Get the list of ratios
        ratio_sales = list(ratio_dict_sales.values())
        ratio_prices = list(ratio_dict_prices.values())

        # Get the list of locations
        locations = list(ratio_dict_sales.keys())

        # Clamp the ratios
        ratio_sales = self.clamp(ratio_sales)
        ratio_prices = self.clamp(ratio_prices)

        # Normalize the ratios
        normalized_ratio_sales = self.normalize(ratio_sales)
        normalized_ratio_prices = self.normalize(ratio_prices)
       
        # Get the colormap
        cmap = plt.get_cmap("plasma")

        # Get the colors for the ratios
        ratio_colors_sales = cmap(normalized_ratio_sales)
        ratio_colors_prices = cmap(normalized_ratio_prices)

        # Initialize the color column in the maps
        map_sales["color"] = "grey"
        map_prices["color"] = "grey"

        # Set the color for each location in the maps
        for location, color in zip(locations, ratio_colors_sales):
            hex_color = mcolors.rgb2hex(color)
            map_sales.loc[map_sales["lad11nm"] == location, "color"] = hex_color
        
        for location, color in zip(locations, ratio_colors_prices):
            hex_color = mcolors.rgb2hex(color)
            map_prices.loc[map_prices["lad11nm"] == location, "color"] = hex_color

        # Create a legend for the colors
        color_index = [Line2D([0], [0], color=cmap(0.), lw=4), Line2D([0], [0], color=cmap(.5), lw=4), Line2D([0], [0], color=cmap(1.), lw=4), Line2D([0], [0], color="grey", lw=4)]
        
        # Create a subplot for sales and prices
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the maps
        for i, ds in enumerate(["ds6", "ds9"]):
            map_sales.plot(color=map_sales["color"], ax=axs[0])
            axs[0].set_title("Sales volume growth")

            map_prices.plot(color=map_prices["color"], ax=axs[1])
            axs[1].set_title("Median price growth")
            
            # Add the legend to the plot
            axs[1].legend(color_index, ["Low", "Medium", "High", "Missing data"])

            # Remove the ticks from the plots
            axs[0].set_xticks([])
            axs[0].set_yticks([])
            axs[1].set_xticks([])
            axs[1].set_yticks([])

        # Show the plot
        plt.show()
           
# Main function to run the program
def main():
    building_type = sys.argv[1]
    map = Map(building_type)
    
    map_file = map.open_map()
    map.plot_map(map_file)
    
# Run the main function
if __name__ == "__main__":
    main()
