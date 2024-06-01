import pandas as pd
import numpy as np
import os
pd.set_option("future.no_silent_downcasting", True)

# Define a class to process data
class DataProcessor:
    def __init__(self, xls_spreadsheet, derived_spreadsheet, augmentation_spreadsheets):
        # Initialize the class with the spreadsheet and augmentation files
        self.xls_spreadsheet = xls_spreadsheet
        self.derived_spreadsheet = derived_spreadsheet
        self.augmentation_spreadsheets = augmentation_spreadsheets
    
    def transform_and_interpolate(self, df, has_regions_col, has_locations_col):
        # Transform and interpolate the data
        regions = df["Region"]
        df = df.drop(columns="Region")
        col_to_keep = (regions, "Region")

        if has_locations_col:
            locations = df["Location"]
            df = df.drop(columns="Location")
            col_to_keep = (locations, "Location")

        df.columns = pd.to_datetime(df.columns, format="%d/%m/%Y")
        df = df.T
        df = df.apply(pd.to_numeric, errors="coerce")
        df_monthly = df.resample("MS").interpolate(method="linear")
        df_monthly = df_monthly.T
        df_monthly.insert(0, col_to_keep[1], col_to_keep[0])
        df_monthly.set_index(col_to_keep[1], inplace=True)
        
        if has_locations_col:
            return df_monthly, regions, locations

        return df_monthly, regions
        

    def augment_data(self, df, region):
        # Augment the data with additional features
        df["Date"] = pd.to_datetime(df["Date"])

        df_gdp = pd.ExcelFile(f"{self.augmentation_spreadsheets}/gdp.xlsx").parse("Sheet1") #gdp is yearly
        df_population = pd.ExcelFile(f"{self.augmentation_spreadsheets}/population.xlsx").parse("Sheet1") #population is yearly
        df_inflation = pd.ExcelFile(f"{self.augmentation_spreadsheets}/inflation.xlsx").parse("Sheet1") #inflation is monthly
        df_interest = pd.ExcelFile(f"{self.augmentation_spreadsheets}/interest.xlsx").parse("Sheet1") #interest is quarterly
        
        complicated_augmentations = [df_gdp, df_population]
        # Interpolate to monthly intervals
        for i, new_df in enumerate(complicated_augmentations):         
            new_df_monthly = self.transform_and_interpolate(new_df, True, False)[0]
            try:
                df = pd.concat([df, new_df_monthly.loc[region]], axis=1)
            except:
                df = pd.concat([df, new_df_monthly.loc["England and Wales"]], axis=1)        
            if i == 0:
                df.rename(columns={df.columns[-1]: "GDP"}, inplace=True)
            else:
                df.rename(columns={df.columns[-1]: "Population"}, inplace=True)
       
        simple_augmentations = [df_inflation, df_interest]
        for new_df in simple_augmentations:
            new_df["Date"] = pd.to_datetime(new_df["Date"])
            new_df.set_index("Date", inplace=True)
            new_df_monthly = new_df.resample("MS").interpolate(method="linear")
            df = pd.concat([df, new_df_monthly], axis=1)

        df = df.drop(df.index[[0, -3, -2, -1]])
        return df

    def preprocess_data(self):
        # Preprocess the data
        num = self.xls_spreadsheet[7:11] # Take ds# from the path
        spreadsheet = pd.ExcelFile(self.xls_spreadsheet)
        sheets = spreadsheet.sheet_names
        sheets[:] = sheets[5:len(sheets) - 1]

        directory = f"{self.derived_spreadsheet}"
        os.makedirs(directory, exist_ok=True)

        for i in range(len(sheets)):            
            df = spreadsheet.parse(sheets[i])
            df = df.drop([0, 1, 2, 3, 4])
            df = df.reset_index(drop=True)

            df.replace("", np.nan, inplace=True)

            dates3M = pd.date_range(start="12/1/1995", end="3/1/2023", freq="3MS")
            dates3M_str = dates3M.strftime("%d/%m/%Y")

            dates1M = pd.date_range(start="12/1/1995", end="3/1/2023", freq="MS")
            dates1M_str = dates1M.strftime("%d/%m/%Y")

            if sheets[i].startswith(("1", "4")):
                df = df.drop(df.columns[[0]], axis=1)
                df.rename(columns={df.columns[0]: "Region"}, inplace=True)
                df.columns = ["Region"] + dates3M_str.tolist()

                df, regions = self.transform_and_interpolate(df, True, False)

                for region in regions:
                    os.makedirs(f"{directory}/{num}/{sheets[i]}/", exist_ok=True)
                    df_region = df.loc[region]
                    
                    df_region = df_region.to_frame()
                    df_region.rename(columns={df_region.columns[0]: "Statistic"}, inplace=True)
                    df_region.insert(0, "Date", dates1M_str.tolist())

                    df_region = self.augment_data(df_region, region)

                    df_region.reset_index()
                    df_region.to_csv(f"{directory}/{num}/{sheets[i]}/{region}.csv", index=False)

            else:
                df = df.drop(df.columns[[0, 2]], axis=1)
                df.rename(columns={df.columns[0]: "Region", df.columns[1]: "Location"}, inplace=True)
                df.columns = ["Region", "Location"] + dates3M_str.tolist()

                # Interpolate to monthly intervals
                df, regions, locations = self.transform_and_interpolate(df, True, True)

                for j, location in enumerate(locations):
                    region = regions[j]
                    os.makedirs(f"{directory}/{num}/{sheets[i]}/", exist_ok=True)
                    df_location = df.loc[location]

                    df_location = df_location.to_frame()
                    df_location.rename(columns={df_location.columns[0]: "Statistic"}, inplace=True)
                    df_location.insert(0, "Date", dates1M_str.tolist())

                    df_location = self.augment_data(df_location, region)

                    df_location.reset_index()
                    df_location.to_csv(f"{directory}/{num}/{sheets[i]}/{location}.csv", index=False)

class LocationParser:
    def __init__(self, xls_spreadsheet):
        # Initialize the class with the spreadsheet file
        self.xls_spreadsheet = xls_spreadsheet

    def load_spreadsheet(self):
        # Load the spreadsheet into a pandas DataFrame
        spreadsheet = pd.ExcelFile(self.xls_spreadsheet)
        return spreadsheet

    def parse_locations(self):
        # Parse the locations from the spreadsheet
        spreadsheet = self.load_spreadsheet()
        useful_pages = {"1a":"Regions", "2a":"LocalAuthorities", "3a":"Counties", "4a":"CombinedAuthorities"}
        for page in useful_pages.keys():
            df = spreadsheet.parse(page)
            if page == "1a" or page == "4a":
                locations = df[df.columns[1]]
            else:
                locations = df[df.columns[1]] + ":" + df[df.columns[3]]
            
            locations = locations[5:]
            text_file = useful_pages[page] + ".txt"
                        
            directory = f"Dataset/Derived/Text/{text_file}"

            with open(directory, "w+") as f:
                for location in locations:
                    f.write(f"{str(location)}\n")

def main():
    # Create the necessary directories
    os.makedirs("Dataset/Derived/", exist_ok=True)
    os.makedirs("Dataset/Derived/", exist_ok=True)
    os.makedirs("Dataset/Derived/Text/", exist_ok=True)

    # Process the data
    processor6 = DataProcessor("Dataset/ds6_number_of_properties_sold.xls", "Dataset/Derived", "Dataset/Additional")
    processor6.preprocess_data()

    processor9 = DataProcessor("Dataset/ds9_median_price.xls", "Dataset/Derived", "Dataset/Additional")
    processor9.preprocess_data()

    # Parse the locations
    parser = LocationParser("Dataset/ds9_median_price.xls")
    parser.parse_locations()

if __name__ == "__main__":
    main()