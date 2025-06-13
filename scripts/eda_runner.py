import pandas as pd
import os, sys
from datetime import datetime

# Add scripts to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from load_data import load_data
from data_processing import DataProcessing
from data_visualization import DataVisualizer

# Load the data
data = load_data("data/raw/MachineLearningRating_v3.zip", filename="MachineLearningRating_v3.txt")

# Initialize data processing
processor = DataProcessing(data)

# Handle missing data
high_missing = ['NumberOfVehiclesInFleet', 'CrossBorder', 'CustomValueEstimate', 'Converted', 'Rebuilt', 'WrittenOff']
moderate_missing = ['NewVehicle', 'Bank', 'AccountType']
low_missing = ['Gender', 'MaritalStatus', 'Cylinders', 'cubiccapacity', 'kilowatts', 
               'NumberOfDoors', 'VehicleIntroDate', 'Model', 'make', 'VehicleType', 
               'mmcode', 'bodytype', 'CapitalOutstanding']

data = processor.handle_missing_data('high', high_missing)
data = processor.handle_missing_data('moderate', moderate_missing)
data = processor.handle_missing_data('low', low_missing)

# Convert dates
data['VehicleIntroDate'] = pd.to_datetime(data['VehicleIntroDate'], format='%d/%m/%Y', errors='coerce')

# Initialize visualizer
vis = DataVisualizer(data)

# Run univariate analysis
numerical_cols = ['SumInsured', 'CalculatedPremiumPerTerm', 'TotalPremium', 'TotalClaims']
categorical_cols = ['LegalType', 'Bank', 'AccountType', 'MaritalStatus', 'Gender', 
                    'Province', 'VehicleType', 'AlarmImmobiliser', 'TrackingDevice', 'Product']

vis.univariate_analysis(numerical_cols, categorical_cols)

# Plot cover type frequencies
cover_type_counts = data['CoverType'].value_counts()
cover_type_counts.plot(kind='bar', figsize=(12, 4), title='Cover Type Frequencies')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Geographical trends
common_cover_types = [
    'Own Damage', 'Passenger Liability', 'Windscreen', 'Third Party', 
    'Keys and Alarms', 'Signage and Vehicle Wraps', 'Emergency Charges', 
    'Cleaning and Removal of Accidental Debris'
]
vis.plot_geographical_trends(common_cover_types)

# Cap outliers
df_capped = vis.cap_all_outliers(numerical_cols)

# Re-visualize outliers
vis1 = DataVisualizer(df_capped)
vis1.plot_outliers_boxplot(numerical_cols)

# Save cleaned data
os.makedirs("data/cleaned", exist_ok=True)
df_capped.to_csv("data/cleaned/cleaned_data.csv", index=False)
