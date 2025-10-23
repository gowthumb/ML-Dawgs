# ML-Dawgs
SC4000 Project 

The files device_gnss.csv and device_imu.csv are from smartphone-decimeter-2022/train/2020-05-15-US-MTV-1/GooglePixel4XL/

📡 GNSS–IMU Data Merger and Visualizer
🧩 Overview

This script merges GNSS (Global Navigation Satellite System) and IMU (Inertial Measurement Unit) datasets based on synchronized timestamps, enabling you to analyze a device’s motion and position together in a unified timeline.

It combines sensor measurements (acceleration, gyroscope, magnetometer) with geolocation data (latitude, longitude, altitude, and ECEF coordinates).
After merging, the script visualizes the trajectory of the device in 3D space.

🗂️ Input Files

You need two CSV files:

device_gnss.csv
Contains GNSS data such as:

Timestamp (utcTimeMillis)

Position (WlsPositionXEcefMeters, WlsPositionYEcefMeters, WlsPositionZEcefMeters)

Other satellite-derived info (velocity, uncertainty, etc.)

device_imu.csv
Contains IMU sensor readings such as:

Timestamp (utcTimeMillis)

Message type (UncalAccel, UncalGyro, UncalMag)

Measurement values (acceleration, rotation rate, magnetic field)

⚙️ How It Works

Load Data
Both CSVs are read using pandas.read_csv() into DataFrames.
Timestamp Alignment
The IMU and GNSS data are merged based on utcTimeMillis, so each IMU reading gets matched to the nearest GNSS fix (position).
Data Schema Normalization
The code standardizes columns, ensuring consistent naming across both datasets.
Columns from GNSS data are prefixed with Wls... for clarity.
Columns from IMU data include MessageType_x (e.g., UncalAccel, UncalGyro) and measurement axes.

Merging

A “nearest join” aligns IMU samples to the closest GNSS timestamp, preserving all available IMU samples.

Preview Output

The merged DataFrame prints the first 5 rows with all 54 combined columns.

📊 Output Example

A snippet of the merged dataset looks like this:

MessageType_x	utcTimeMillis	MeasurementX	...	WlsPositionXEcefMeters	WlsPositionYEcefMeters	WlsPositionZEcefMeters
UncalAccel	1589573679447	0.237913	...	-2.693907e+06	-4.297452e+06	3.854203e+06
UncalGyro	1589573679450	-0.022198	...	-2.693907e+06	-4.297452e+06	3.854203e+06

Each row now includes both IMU motion and GNSS position at that moment in time.

🧭 Visualization

After merging, the script generates a 3D trajectory plot of the device’s motion using matplotlib.
X, Y, Z axes: ECEF (Earth-Centered, Earth-Fixed) coordinates in meters.
Each point: Represents the device’s estimated position at a given time.
Path shape: Shows the route or motion pattern captured by the GNSS receiver.

Example:
If you walked in a circle with your phone, the 3D plot would show a curved path representing that circular movement.

📈 Output Graph

The graph typically displays:
A 3D scatter line showing the path traced by GNSS over time.
The line color or animation (if added) can correspond to timestamp progression or IMU activity intensity.

🚀 How to Run
Place your two CSV files (device_gnss.csv and device_imu.csv) in the same directory as the script.

Run:

python merge_gnss_imu.py


The script will:
Print a preview of the merged dataset.
Display the 3D trajectory plot.

💡 Applications

Analyzing motion patterns in pedestrian or vehicle navigation.
Synchronizing GNSS and IMU data for sensor fusion.
Debugging mobile data logs or location-based systems.
Building datasets for ML models in localization.
