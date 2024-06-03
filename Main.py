import subprocess
import os
import tkinter as tk 
from tkinter import ttk
from ttkwidgets.autocomplete import AutocompleteCombobox
import tkcalendar as cal
import datetime

# Define the main class for the application
class Main:
    def __init__(self, root):
        self.root = root
        self.root.geometry("480x480")  # Set window size
        self.selected_date = datetime.date.today().strftime("%d/%m/%Y") # Get today's date
        self.initialize_files()  # Initialize file paths
        self.initialize_dropdown_dicts()  # Initialize dropdown dictionaries
        self.create_window_elements()  # Create GUI elements
        self.check_dataset_path()  # Check if dataset path exists

    def initialize_files(self):
        # Initialize the file paths
        self.local_authorities_file = "Dataset/Derived/Text/LocalAuthorities.txt"
        self.counties_file = "Dataset/Derived/Text/Counties.txt"
        self.combined_authorities_file = "Dataset/Derived/Text/CombinedAuthorities.txt"
        self.regions_file = "Dataset/Derived/Text/Regions.txt"
        self.dataset_path = "Dataset/Derived/"

    def initialize_dropdown_dicts(self):
        # Initialize the dictionaries for the dropdown menus
        self.datasets_d = {"Number of Properties Sold": "ds6", "Median Price": "ds9"}
        self.divisions_d = {"Regions": "1", "Local Authorities": "2", "Counties": "3", "Combined Authorities": "4"}
        self.types_d = {"All": "a", "Detached": "b", "Semi-detached": "c", "Terraced": "d", "Flats and Maisonets": "e"}

    def check_dataset_path(self):
        # Check if the dataset path exists, if not, run the Processor.py script
        ds_path = "Dataset/Derived"
        if not os.path.exists(ds_path):
            script_path = "Scripts/Processor.py"
            process = subprocess.Popen(["python", script_path])
            process.wait()
        
    def get_locations(self, file):
        # Get the locations from the file
        with open(file, "r") as f:
            lines = f.readlines()
            if lines:
                if ":" in lines[0]:
                    return {line.strip().split(":")[0]: line.strip().split(":")[1] for line in lines}
                else:
                    return {line.strip(): line.strip() for line in lines}
            else:
                return {} 
            
    def on_division_change(self, event):
        # Handle the event when the division dropdown changes
        division = self.division_dropdown.get()
        regions = self.get_locations(self.regions_file)

        # Update regions dropdown based on the selected division
        match division:
            case "Regions":
                regions = self.get_locations(self.regions_file)

            case "Counties":
                regions = self.get_locations(self.counties_file)

            case "Local Authorities":
                regions = self.get_locations(self.local_authorities_file)

            case "Combined Authorities": 
                regions = self.get_locations(self.combined_authorities_file)            
        
        self.region_dropdown.configure(state="normal")
        if regions:
            self.region_dropdown["values"] = list(regions.keys())
            self.region_dropdown.set_completion_list(list(regions.keys()))  # Update the autocompletion list
            self.region_dropdown.set(list(regions.keys())[0])
            self.location_dropdown.set(list(regions.values())[0])
        else:
            self.region_dropdown["values"] = [""]
            self.region_dropdown.set("")
            self.location_dropdown.set("")
            self.region_dropdown.configure(state="disabled")
        
        self.on_region_change(None)

    def on_region_change(self, event):
        # Handle the event when the region dropdown changes
        division = self.division_dropdown.get()
        region = self.region_dropdown.get()
        small_divisions = {}
        if division == "Local Authorities":
            f = self.local_authorities_file
        else:
            f = self.counties_file
        with open(f, "r") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    reg, small_division = line.split(":")
                    if reg.strip() == region:
                        small_divisions[small_division.strip()] = reg.strip()

        self.location_dropdown.configure(state="normal")
        if small_divisions and not division in ["Regions", "Combined Authorities"]: 
            self.location_dropdown["values"] = list(small_divisions.keys())
            self.location_dropdown.set_completion_list(list(small_divisions.keys()))  # Update the autocompletion list
            self.location_dropdown.set(list(small_divisions.keys())[0])
        else:
            self.location_dropdown["values"] = [""]
            self.location_dropdown.set("")
            self.location_dropdown.configure(state="disabled")
        self.small_divisions = small_divisions

    def create_dropdown(self, options, label_text):
        # Create a dropdown menu
        label = tk.Label(self.root, text=label_text, font=("Georgia", 16))
        label.pack(anchor=tk.W, padx=20)
        
        values = list(options.keys())
        dropdown = AutocompleteCombobox(self.root, values, width=75)
        dropdown.pack(anchor=tk.W, padx=20)
        dropdown.set(values[0])
        
        return dropdown

    def create_calendar(self):
        # Create a calendar widget
        calendar_entry = tk.Entry(self.root)
        today = self.selected_date
        calendar_entry.insert(0, today)
        calendar_entry.bind("<1>", lambda event: self.pick_date(event, calendar_entry))
        calendar_entry.pack(anchor=tk.W, padx=20)

    def create_window_elements(self):
        # Create the window elements
        self.dataset_dropdown = self.create_dropdown(self.datasets_d, "Dataset:")
        self.division_dropdown = self.create_dropdown(self.divisions_d, "Division By:")
        self.division_dropdown.bind("<<ComboboxSelected>>", self.on_division_change)

        self.region_dropdown = self.create_dropdown({"placeholder":""}, "Big division:")
        self.region_dropdown.bind("<<ComboboxSelected>>", self.on_region_change)

        self.location_dropdown = self.create_dropdown({"placeholder":""}, "Small division:")
        self.on_division_change(None)

        self.type_dropdown = self.create_dropdown(self.types_d, "Property type:")

        tk.Label(self.root, text="Predict till:", font=("Georgia", 16)).pack(anchor=tk.W, padx=20)
        self.create_calendar()

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(padx=10, pady=0)

        analyze_button = tk.Button(btn_frame, text="Analyze", command=lambda: self.open_analyzer(), font=("Georgia", 16))
        analyze_button.pack(side=tk.LEFT, padx=10, pady=50)

        predict_button = tk.Button(btn_frame, text="Predict", command=lambda: self.open_prediction(), font=("Georgia", 16))
        predict_button.pack(side=tk.LEFT, padx=10, pady=50)

        map_button = tk.Button(btn_frame, text="Map", command=lambda: self.open_map(), font=("Georgia", 16))
        map_button.pack(side=tk.LEFT, padx=10, pady=50)

    def pick_date(self, event, calendar_entry):
        # Handle the event when a date is picked
        date_window = tk.Toplevel(self.root)
        date_window.title("Select Date")
        
        calendar = cal.Calendar(date_window, selectmode="day", date_pattern="dd/mm/yyyy")
        calendar.pack(padx=20, pady=20)

        def on_date_select():
            selected_date = calendar.get_date()
            calendar_entry.delete(0, tk.END)
            calendar_entry.insert(0, selected_date)
            date_window.destroy()

            self.selected_date = selected_date

        select_button = tk.Button(date_window, text="Select", command=on_date_select)
        select_button.pack(pady=10)

        date_window.bind("<Return>", lambda event: on_date_select())

    def open_analyzer(self):      
        # Open the analyzer script
        dataset = self.datasets_d[self.dataset_dropdown.get()]
        division = self.divisions_d[self.division_dropdown.get()]

        region = self.region_dropdown.get()
        location = self.location_dropdown.get()

        script_path = "Scripts/Analyzer.py"
        subprocess.Popen(["python", script_path, dataset, division, region, location])

    def open_map(self):      
        # Open the map script
        type = self.types_d[self.type_dropdown.get()]

        script_path = "Scripts/Map.py"
        subprocess.Popen(["python", script_path, type])

    def open_prediction(self):
        # Open the prediction script
        dataset = self.datasets_d[self.dataset_dropdown.get()]
        division = self.divisions_d[self.division_dropdown.get()]
        type = self.types_d[self.type_dropdown.get()]

        region = self.region_dropdown.get()
        location = self.location_dropdown.get()

        selected = self.selected_date
        last_in_the_df = "01/12/2022"
        
        month_diff = lambda date1, date2: (12*int(date1[-4:]) + int(date1[3:-5])) - (12*int(date2[-4:]) + int(date2[3:-5]))
        
        window = month_diff(selected, last_in_the_df)

        if window < 0: window = 0

        script_path = "Scripts/Predictioner.py"
        subprocess.Popen(["python", script_path, dataset, division, region, location, type, str(window)])

def main():
    root = tk.Tk()
    root.title("Main menu")
    app = Main(root)
    root.resizable(False, False)
    root.mainloop()

if __name__ == "__main__":
    main()
