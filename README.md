#Land Use and Land Cover (LULC) Classification and Prediction#

##Purpose##
This repository provides a comprehensive framework for land use and land cover (LULC) classification and prediction in Malawi, integrating deep learning and Random Forest approaches:
•	Deep Learning: Python scripts using PyTorch with a ResNet-18 backbone, Focal Loss, SMOTE for class balancing, and spatial dropout for regularization. These scripts classify land use types based on geospatial training points, incorporating spectral indices (e.g., NDVI, NDWI, BSI, EVI) and terrain features (e.g., slope, aspect, elevation).
•	Random Forest: Google Earth Engine (GEE) scripts for Random Forest classification of Landsat imagery for 2010, 2015, 2020, and 2024, generating classified maps, area statistics, and preprocessed data (e.g., Landsat composites, slope, shapefiles) for the deep learning pipeline.
•	Data Preprocessing: GEE scripts export Landsat imagery, DEM-derived features, and shapefiles compatible with the deep learning scripts.
The framework includes:
•	Training Script: Trains a deep learning model to classify land use types (Water=0, Built-up=1, Bareland=2, Agricultural=3, Forest=4) using 2020 data.
•	Prediction Script: Applies the trained model to predict land use classes for 2010, 2015, 2020, or 2024 imagery within a shapefile boundary, generating predictions, area summaries, and visualization maps.
•	GEE Scripts: Preprocess Landsat imagery (2010: Landsat-5; 2015, 2020, 2024: Landsat-8) and generate Random Forest classified maps for validation or as additional features.

###Repository Structure###
•	src/land_use_classifier.py: Trains the deep learning model.
•	src/land_use_predictor.py: Generates predictions using the trained model.
•	src/gee_random_forest_2010.js: Random Forest classification for 2010 Landsat-5 imagery.
•	src/gee_random_forest_2015.js: Random Forest classification for 2015 Landsat-8 imagery.
•	src/gee_random_forest_2020.js: Random Forest classification for 2020 Landsat-8 imagery.
•	src/gee_random_forest_2024.js: Random Forest classification for 2024 Landsat-8 imagery.
•	tests/quick_test.py: Verifies dataset loading and patch extraction for all years.
•	data/sample_training_points.csv: Sample training points (100 points).
•	data/sample_landsat_2010.tif: Sample 2010 Landsat-5 image (256x256 pixels).
•	data/sample_landsat_2015.tif: Sample 2015 Landsat-8 image (256x256 pixels).
•	data/sample_landsat_2020.tif: Sample 2020 Landsat-8 image (256x256 pixels).
•	data/sample_landsat_2024.tif: Sample 2024 Landsat-8 image (256x256 pixels).
•	data/sample_dem.tif: Sample DEM (256x256 pixels).
•	data/sample_slope.tif: Sample slope raster (256x256 pixels).
•	data/sample_malawi.shp: Sample shapefile for Malawi.
•	requirements.txt: Python dependencies.
•	LICENSE: MIT License.

###Setup###
1.	Clone the Repository:
2.	git clone https://github.com/JaneGondwegithub/LULC_classification_and_prediction.git
3.	cd LULC_classification_and_prediction
4.	Install Dependencies:
5.	pip install -r requirements.txt
Ensure PyTorch with CUDA support for GPU usage (see PyTorch installation guide). Install GEE Python API:
earthengine-api>=0.1.357
geemap>=0.32.0
6.	Set Up Google Earth Engine:
o	Sign up at https://earthengine.google.com/ and link to a Google Cloud Project.
o	Authenticate GEE:
o	earthengine-authenticate
o	Test initialization:
o	import ee
o	ee.Initialize()
o	print("GEE initialized successfully!")
7.	Python Version: Use Python 3.8 or higher.
8.	Environment Note: Dependencies like rasterio, geopandas, and earthengine-api may require system libraries (e.g., GDAL). See platform-specific guides if needed.

###Quick-Test Instructions###
To verify the code with sample data:
1.	Run the test script:
2.	python tests/quick_test.py
3.	Expected Output: Prints shapes and labels of the first 5 valid patches for the training dataset (2020) and prediction datasets (2010, 2015, 2020, 2024). Example:
4.	Testing training dataset (2020):
5.	Sample 0: Label=2, Patch shape=torch.Size([17, 32, 32])
6.	Testing prediction dataset (2010):
7.	Sample 0: Coords=(...), Invalid=False, Patch shape=torch.Size([17, 32, 32])
8.	Uses data/sample_training_points.csv, data/sample_landsat_*.tif, data/sample_dem.tif, data/sample_slope.tif, and data/sample_malawi.shp.

###Preprocessing with Google Earth Engine###
To prepare input datasets:
1.	Upload Training Data:
o	Upload Cleaned_Training_Points_2020.csv to GEE Assets (e.g., users/your-username/Training_Points_2020) via the GEE Code Editor (Assets → New → Table).
o	Ensure the CSV has columns latitude, longitude, Class (0=Water, 1=Built-up, 2=Bareland, 3=Agricultural, 4=Forest).
o	Update the training data path in each GEE script to match your GEE username.

2.	Run the GEE Random Forest Scripts:
o	2010 Script:
	Open https://code.earthengine.google.com/b019f2e14f22125a9eb99ab1f7ed20c5 or copy src/gee_random_forest_2010.js to the GEE Code Editor.
	Exports: Landsat_2010_Malawi_Merged_FixedNoData.tif, Classified_Map_2010.tif, slope1.tif, malawi.shp.
o	2015 Script:
	Open https://code.earthengine.google.com/4a7dac9db52b1f5425603f073f833caf or copy src/gee_random_forest_2015.js.
	Exports: Landsat8_2015_Malawi_Merged_FixedNoData.tif, Classified_Map_2015.tif, slope1.tif, malawi.shp.
o	2020 Script:
	Open https://code.earthengine.google.com/0b0e90d07b35ead27132e15ef9a2b083 or copy src/gee_random_forest_2020.js.
	Exports: Landsat8_2020_Malawi_Merged_FixedNoData.tif, Classified_Map_2020.tif, slope1.tif, malawi.shp.
o	2024 Script:
	Open https://code.earthengine.google.com/9ee813a42acf599b69169cb384e12cf2 or copy src/gee_random_forest_2024.js.
	Exports: Landsat8_2024_Malawi_Merged_FixedNoData.tif, Classified_Map_2024.tif, slope1.tif, malawi.shp.
o	Update the training data path (e.g., users/your-username/Training_Points_2020) in each script.
o	Download exports from Google Drive (folder LULC_Malawi) to data/.
3.	Create Sample Data:
o	Crop TIFFs to 256x256 pixels and the shapefile to create sample data:
o	import rasterio
o	from rasterio.windows import Window
o	import geopandas as gpd
o	import pandas as pd

o	# Crop Landsat-2010
o	with rasterio.open("data/Landsat_2010_Malawi_Merged_FixedNoData.tif") as src:
o	    window = Window(0, 0, 256, 256)
o	    data = src.read(window=window)
o	    profile = src.profile
o	    profile.update(width=256, height=256, transform=src.window_transform(window))
o	    with rasterio.open("data/sample_landsat_2010.tif", "w", **profile) as dst:
o	        dst.write(data)

o	# Crop Landsat-2015
o	with rasterio.open("data/Landsat8_2015_Malawi_Merged_FixedNoData.tif") as src:
o	    window = Window(0, 0, 256, 256)
o	    data = src.read(window=window)
o	    profile = src.profile
o	    profile.update(width=256, height=256, transform=src.window_transform(window))
o	    with rasterio.open("data/sample_landsat_2015.tif", "w", **profile) as dst:
o	        dst.write(data)

o	# Crop Landsat-2020
o	with rasterio.open("data/Landsat8_2020_Malawi_Merged_FixedNoData.tif") as src:
o	    window = Window(0, 0, 256, 256)
o	    data = src.read(window=window)
o	    profile = src.profile
o	    profile.update(width=256, height=256, transform=src.window_transform(window))
o	    with rasterio.open("data/sample_landsat_2020.tif", "w", **profile) as dst:
o	        dst.write(data)

o	# Crop Landsat-2024
o	with rasterio.open("data/Landsat8_2024_Malawi_Merged_FixedNoData.tif") as src:
o	    window = Window(0, 0, 256, 256)
o	    data = src.read(window=window)
o	    profile = src.profile
o	    profile.update(width=256, height=256, transform=src.window_transform(window))
o	    with rasterio.open("data/sample_landsat_2024.tif", "w", **profile) as dst:
o	        dst.write(data)

o	# Crop DEM
o	with rasterio.open("data/Raster.tif") as src:
o	    window = Window(0, 0, 256, 256)
o	    data = src.read(window=window)
o	    profile = src.profile
o	    profile.update(width=256, height=256, transform=src.window_transform(window))
o	    with rasterio.open("data/sample_dem.tif", "w", **profile) as dst:
o	        dst.write(data)

o	# Crop slope
o	with rasterio.open("data/slope1.tif") as src:
o	    window = Window(0, 0, 256, 256)
o	    data = src.read(window=window)
o	    profile = src.profile
o	    profile.update(width=256, height=256, transform=src.window_transform(window))
o	    with rasterio.open("data/sample_slope.tif", "w", **profile) as dst:
o	        dst.write(data)

o	# Crop shapefile
o	gdf = gpd.read_file("data/malawi.shp")
o	bbox = gpd.GeoDataFrame(geometry=[gdf.geometry.bounds]).set_crs("EPSG:4326")
o	sample_gdf = gdf.clip(bbox)
o	sample_gdf.to_file("data/sample_malawi.shp")

o	# Sample training points
o	df = pd.read_csv("data/Cleaned_Training_Points_2020.csv")
o	sample_df = df.head(100)
o	sample_df.to_csv("data/sample_training_points.csv", index=False)
4.	Add to Git:
5.	git add data/sample_landsat_2010.tif data/sample_landsat_2015.tif data/sample_landsat_2020.tif data/sample_landsat_2024.tif data/sample_dem.tif data/sample_slope.tif data/sample_malawi.shp data/sample_training_points.csv
6.	git commit -m "Add sample data for 2010, 2015, 2020, 2024"
7.	git push origin main

###Running the Training Script###
To train the deep learning model:
1.	Obtain Full Datasets:
o	Landsat-8 Image (2020): Download Landsat8_2020_Malawi_Merged_FixedNoData.tif using the 2020 GEE script or from USGS EarthExplorer (4 bands: Blue, Green, Red, NIR).
o	DEM: Download Raster.tif from SRTM or your data provider. Ensure it aligns with the 2020 Landsat-8 image.
o	Training Points: Use Cleaned_Training_Points_2020.csv with columns latitude, longitude, Class (0=Water, 1=Built-up, 2=Bareland, 3=Agricultural, 4=Forest).
o	Place files in data/ or update paths in src/land_use_classifier.py (e.g., train_csv_path, image_path, dem_path).
2.	Run:
3.	python src/land_use_classifier.py
4.	Outputs:
o	Model weights: logs/best_land_use_forecaster_v24_final.pth
o	Logs: logs/forecast_training_log_v24_*.log
o	Plots (loss, accuracy, misclassification map): logs/
o	Validation points: logs/validation_points_v24_final.csv

###Running the Prediction Script###
To predict land use classes:
1.	Obtain Full Datasets:
o	Landsat Images: Download Landsat_2010_Malawi_Merged_FixedNoData.tif, Landsat8_2015_Malawi_Merged_FixedNoData.tif, Landsat8_2020_Malawi_Merged_FixedNoData.tif, Landsat8_2024_Malawi_Merged_FixedNoData.tif using GEE scripts or USGS EarthExplorer (4 bands each).
o	Slope: Download slope1.tif using GEE scripts. Ensure it aligns with Landsat images.
o	Shapefile: Use malawi.shp to define the prediction area.
o	Trained Model: Ensure logs/best_land_use_forecaster_v24_final.pth exists.
o	Place files in data/ or update paths in src/land_use_predictor.py (e.g., image_path_2024, slope_path, shp_path).
2.	Run:
3.	python src/land_use_predictor.py
4.	Outputs:
o	Predictions: logs/predictions_2024_v24_final.csv (latitude, longitude, predicted class, confidence, invalid patch flag)
o	Summary: logs/predictions_2024_summary_v24_final.csv (class counts and areas in hectares/km²)
o	Map: logs/predictions_map_2024_v24_final.png

###Running the Random Forest Classifier###
To generate Random Forest classified maps:
1.	Run GEE Scripts:
o	2010: Open https://code.earthengine.google.com/b019f2e14f22125a9eb99ab1f7ed20c5 or copy src/gee_random_forest_2010.js.
o	2015: Open https://code.earthengine.google.com/4a7dac9db52b1f5425603f073f833caf or copy src/gee_random_forest_2015.js.
o	2020: Open https://code.earthengine.google.com/0b0e90d07b35ead27132e15ef9a2b083 or copy src/gee_random_forest_2020.js.
o	2024: Open https://code.earthengine.google.com/9ee813a42acf599b69169cb384e12cf2 or copy src/gee_random_forest_2024.js.
o	Update the training data path (e.g., users/your-username/Training_Points_2020).
o	Export Classified_Map_2010.tif, Classified_Map_2015.tif, Classified_Map_2020.tif, Classified_Map_2024.tif, Landsat images, slope1.tif, and malawi.shp to Google Drive (folder LULC_Malawi).
2.	Integration with Deep Learning:
o	Add Random Forest classified maps as extra bands in land_use_predictor.py:
o	import rasterio
o	import numpy as np

o	# In LandUsePredictionDataset
o	with rasterio.open("data/Classified_Map_2010.tif") as src:
o	    rf_band_2010 = src.read(1, window=window)
o	    rf_band_2010 = np.where(rf_band_2010 == src.nodata, 0, rf_band_2010)
o	with rasterio.open("data/Classified_Map_2015.tif") as src:
o	    rf_band_2015 = src.read(1, window=window)
o	    rf_band_2015 = np.where(rf_band_2015 == src.nodata, 0, rf_band_2015)
o	with rasterio.open("data/Classified_Map_2020.tif") as src:
o	    rf_band_2020 = src.read(1, window=window)
o	    rf_band_2020 = np.where(rf_band_2020 == src.nodata, 0, rf_band_2020)
o	with rasterio.open("data/Classified_Map_2024.tif") as src:
o	    rf_band_2024 = src.read(1, window=window)
o	    rf_band_2024 = np.where(rf_band_2024 == src.nodata, 0, rf_band_2024)
o	image = np.concatenate([image, rf_band_2010[np.newaxis, :, :], rf_band_2015[np.newaxis, :, :], rf_band_2020[np.newaxis, :, :], rf_band_2024[np.newaxis, :, :]], axis=0)
o	Update config["num_channels"] to 21 in src/land_use_predictor.py to account for the 4 additional Random Forest bands.
3.	Compare Results:
o	Compare Random Forest and deep learning predictions:
o	import pandas as pd
o	import rasterio
o	import numpy as np

o	# Load deep learning predictions
o	dl_preds = pd.read_csv("logs/predictions_2024_v24_final.csv")
o	# Load Random Forest classified map (e.g., 2024)
o	with rasterio.open("data/Classified_Map_2024.tif") as src:
o	    rf_data = src.read(1)
o	# Compare (align coordinates as needed)

###Notes###
•	Sample Data: The data/ directory contains small subsets for testing (e.g., 100 training points, 256x256 pixel rasters). Full datasets are required for training and prediction.
•	CRS Compatibility: All scripts assume EPSG:4326. Adjust LandUseDataset and LandUsePredictionDataset classes if your data uses a different CRS.
•	Performance: Deep learning scripts use GPU (CUDA) if available, else CPU (requires ~16GB RAM, 4GB VRAM). GEE exports may take time for large areas.
•	Large Files: Full TIFFs and shapefiles are not included due to size constraints. Use GEE scripts or USGS/SRTM sources to obtain them.
•	Class Labels: Classes are Water (0), Built-up (1), Bareland (2), Agricultural (3), Forest (4).
•	GEE Script Access: Random Forest scripts are available in the repository (src/gee_random_forest_*.js) and via GEE links:
o	2010: https://code.earthengine.google.com/b019f2e14f22125a9eb99ab1f7ed20c5
o	2015: https://code.earthengine.google.com/4a7dac9db52b1f5425603f073f833caf
o	2020: https://code.earthengine.google.com/0b0e90d07b35ead27132e15ef9a2b083
o	2024: https://code.earthengine.google.com/9ee813a42acf599b69169cb384e12cf2
•	Training Data: The GEE scripts use users/your-username/Training_Points_2020. Update to your GEE username. Using 2020 training data for 2010 and 2015 may reduce accuracy; consider year-specific datasets if available.
•	Customization: Modify config dictionaries in src/land_use_classifier.py and src/land_use_predictor.py for hyperparameters (e.g., patch_size, batch_size, grid_step).

###Computer Code Availability###
Source code, quick-test script, GEE scripts, and sample data are available at https://github.com/JaneGondwegithub/LULC_classification_and_prediction. The repository is public and allows anonymous access.

###License###
Copyright (c) 2025 Jane Ferah Gondwe.
This project is licensed under the MIT License - see the LICENSE file for details.

