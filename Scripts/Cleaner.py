import pandas as pd
import numpy as np
import os
pd.set_option("future.no_silent_downcasting", True)

class DataCleaner:
    def __init__(self, xls_spreadsheet, derived_spreadsheet):
        self.xls_spreadsheet = xls_spreadsheet
        self.derived_spreadsheet = derived_spreadsheet

    def load_big_spreadsheet(self):
        spreadsheet = pd.ExcelFile(self.xls_spreadsheet)
        print("Spreadsheet loaded")
        return spreadsheet

    def clean_data(self):
        spreadsheet = self.load_big_spreadsheet()
        sheets = spreadsheet.sheet_names
        sheets[:] = sheets[5:len(sheets) - 1]

        directory = f"{self.derived_spreadsheet}"
        os.makedirs(directory, exist_ok=True)

        for i in range(len(sheets)):
            df = spreadsheet.parse(sheets[i])
            df = df.drop([0, 1, 2, 3, 4])
            df = df.reset_index(drop=True)

            df.replace("", np.nan, inplace=True)

            dates = pd.date_range(start="12/1/1995", end="3/1/2023", freq="3MS")
            dates_str = dates.strftime("%d/%m/%Y")

            if sheets[i].startswith(("1", "4")):
                df = df.drop(df.columns[[0]], axis=1)
                df.rename(columns={df.columns[0]: "Region"}, inplace=True)
                df.columns = ["Region"] + dates_str.tolist()

                region = df["Region"]
                df = df.drop(columns="Region")
                df.columns = pd.to_datetime(df.columns, format="%d/%m/%Y")

                df = df.T
                df = df.apply(pd.to_numeric, errors="coerce")

                df = df.resample("MS").interpolate(method="linear")
                df = df.T
                df.insert(0, "Region", region)

                df = df.drop(df.columns[[1, -3, -2, -1]], axis=1)

            elif sheets[i].startswith(("2", "3")):
                df = df.drop(df.columns[[0, 2]], axis=1)
                if sheets[i].startswith("2"):
                    df.rename(columns={df.columns[0]: "Region", df.columns[1]: "Local Authority"}, inplace=True)
                    df.columns = ["Region", "Local Authority"] + dates_str.tolist()
                    region = df["Region"]
                    location = df["Local Authority"]
                    df = df.drop(columns="Region")
                    df = df.drop(columns="Local Authority")

                else:
                    df.rename(columns={df.columns[0]: "Region", df.columns[1]: "County"}, inplace=True)
                    df.columns = ["Region", "County"] + dates_str.tolist()
                    region = df["Region"]
                    location = df["County"]
                    df = df.drop(columns="Region")
                    df = df.drop(columns="County")

                df.columns = pd.to_datetime(df.columns, format="%d/%m/%Y")

                df = df.T
                df = df.apply(pd.to_numeric, errors="coerce")
                df = df.resample("MS").interpolate(method="linear")

                df = df.T
                df.insert(0, "Region", region)

                if sheets[i].startswith("2"):
                    df.insert(0, "Local Authority", location)

                elif sheets[i].startswith("3"):
                    df.insert(0, "County", location)

                df = df.drop(df.columns[[2, -3, -2, -1]], axis=1)

            df.to_csv(f"{directory}/{sheets[i]}.csv", index=False)

        print(f"CSVs have been created for all the sheets in dataset {self.derived_spreadsheet[-3:-1]}\nYou can find them in {directory}\n")

class LocationsParser:
    def __init__(self, csv_spreadsheet, text_file):
        self.csv_spreadsheet = "Dataset/Derived/9/" + csv_spreadsheet
        self.text_file = text_file

    def load_little_spreadsheet(self):
        spreadsheet = pd.read_csv(self.csv_spreadsheet)
        return spreadsheet

    def parse_locations(self):
        spreadsheet = self.load_little_spreadsheet()
        i = len("Dataset/Derived/9/")
        if self.csv_spreadsheet[i] == "1" or self.csv_spreadsheet[i] == "4":
            locations = spreadsheet[spreadsheet.columns[0]]
        else:
            locations = spreadsheet[spreadsheet.columns[1]] + ":" + spreadsheet[spreadsheet.columns[0]]

        directory = os.path.dirname(self.text_file)
        os.makedirs(directory, exist_ok=True)

        with open(self.text_file, "w+") as f:
            for location in locations[1:]:
                f.write(f"{str(location)}\n")

        print(f"All possible small region names were parsed to {self.text_file}")

def main():
    os.makedirs("Dataset/Derived/6/", exist_ok=True)
    os.makedirs("Dataset/Derived/9/", exist_ok=True)
    os.makedirs("Dataset/Derived/Text/", exist_ok=True)

    cleaner6 = DataCleaner("Dataset/ds6_number_of_properties_sold.xls", "Dataset/Derived/6/")
    cleaner6.clean_data()

    cleaner9 = DataCleaner("Dataset/ds9_median_price.xls", "Dataset/Derived/9/")
    cleaner9.clean_data()

    parser_regions = LocationsParser("1a.csv", "Dataset/Derived/Text/Regions.txt")
    parser_regions.parse_locations()

    parser_local_authorities = LocationsParser("2a.csv", "Dataset/Derived/Text/LocalAuthorities.txt")
    parser_local_authorities.parse_locations()

    parser_counties = LocationsParser("3a.csv", "Dataset/Derived/Text/Counties.txt")
    parser_counties.parse_locations()

    parser_combined_authorities = LocationsParser("4a.csv", "Dataset/Derived/Text/CombinedAuthorities.txt")
    parser_combined_authorities.parse_locations()

if __name__ == "__main__":
    main()