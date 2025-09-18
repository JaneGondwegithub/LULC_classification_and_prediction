#Land Use and Land Cover (LULC) Classification and Prediction#

##Purpose##
This repository contains Python scripts for land use and land cover (LULC) classification and prediction in Malawi using Landsat-8 imagery and Digital Elevation Model (DEM) data. The framework is built with PyTorch, leveraging a ResNet-18 backbone, Focal Loss, SMOTE for class balancing, and spatial dropout for regularization. It includes:
•	Training Script: Trains a model to classify land use types based on geospatial training points, incorporating spectral indices (e.g., NDVI, NDWI) and terrain features (e.g., slope, aspect, TRI).
•	Prediction Script: Applies the trained model to predict land use classes for new imagery (e.g., 2024 Landsat-8 data) within a shapefile boundary, generating predictions, area summaries, and a visualization map.
Note: The repository description mentions integration of random forest and deep learning approaches. Currently, only the deep learning component is included. Random forest integration may be planned for future updates.

###Repository Structure###
•	src/land_use_classifier.py: Main script for training the LULC classification model.
•	src/land_use_predictor.py: Script for generating predictions using the trained model.
•	tests/quick_test.py: Quick-test script to verify dataset loading and patch extraction.
•	data/sample_training_points.csv: Sample training points (100 points, subset of full dataset).
•	data/sample_landsat_2020.tif: Sample 2020 Landsat-8 image (cropped to 256x256 pixels).
•	data/sample_dem.tif: Sample DEM (cropped to 256x256 pixels).
•	data/sample_landsat_2024.tif: Sample 2024 Landsat-8 image (cropped to 256x256 pixels).
•	data/sample_slope.tif: Sample slope raster (cropped to 256x256 pixels).
•	data/sample_malawi.shp: Sample shapefile defining the prediction area.
•	requirements.txt: Python dependencies.
•	LICENSE: MIT License.

###Setup###
1.	Clone the Repository:
2.	git clone https://github.com/JaneGondwegithub/LULC_classification_and_prediction.git
3.	cd LULC_classification_and_prediction
4.	Install Dependencies:
5.	pip install -r requirements.txt
Ensure you have PyTorch with CUDA support for GPU usage (see PyTorch installation guide).
6.	Python Version: Use Python 3.8 or higher.
7.	Environment Note: Dependencies like rasterio and geopandas may require system libraries (e.g., GDAL). See platform-specific installation guides if needed.

###Quick-Test Instructions###
To verify the code works with the provided sample data:
1.	Run the test script:
2.	python tests/quick_test.py
3.	Expected Output: Prints the shape and label of the first 5 valid patches from the sample training dataset (e.g., Sample 0: Label=2, Patch shape=torch.Size([17, 64, 64])).
4.	The test uses data/sample_training_points.csv, data/sample_landsat_2020.tif, and data/sample_dem.tif to load and process patches.

###Running the Training Script###
To train the model with the full datasets:
1.	Obtain Full Datasets:
o	Landsat-8 Image (2020): Download Landsat8_2020_Malawi_Merged_FixedNoData.tif from USGS EarthExplorer. Ensure it has 4 bands and minimal NoData values.
o	DEM: Download Raster.tif from SRTM or your data provider. Ensure it aligns spatially with the 2020 Landsat-8 image.
o	Training Points: Use the full Cleaned_Training_Points_2020.csv, with columns latitude, longitude, and Class (integers 0-4). The structure matches data/sample_training_points.csv.
o	Place these files in the data/ directory or update paths in src/land_use_classifier.py (e.g., train_csv_path, image_path, dem_path).
2.	Run the Training Script:
3.	python src/land_use_classifier.py
4.	Outputs:
o	Model weights: logs/best_land_use_forecaster_v24_final.pth.
o	Training logs: logs/forecast_training_log_v24_*.log.
o	Plots (loss, accuracy, misclassification map): logs/.
o	Validation points: logs/validation_points_v24_final.csv.

###Running the Prediction Script###
To generate predictions using the trained model:
1.	Obtain Full Datasets:
o	Landsat-8 Image (2024): Download Landsat8_2024_Malawi_Merged_FixedNoData.tif from USGS EarthExplorer. Ensure it has 4 bands.
o	Slope Raster: Download or generate slope1.tif (derived from DEM). Ensure it aligns with the 2024 Landsat-8 image.
o	Shapefile: Use malawi.shp to define the prediction area. The structure matches data/sample_malawi.shp.
o	Place these files in the data/ directory or update paths in src/land_use_predictor.py (e.g., image_path_2024, slope_path, shp_path).
2.	Ensure Trained Model: Verify that logs/best_land_use_forecaster_v24_final.pth exists from the training step.
3.	Run the Prediction Script:
4.	python src/land_use_predictor.py
5.	Outputs:
o	Predictions: logs/predictions_2024_v24_final.csv (latitude, longitude, predicted class, confidence, invalid patch flag).
o	Summary: logs/predictions_2024_summary_v24_final.csv (class counts and areas in hectares/km²).
o	Prediction map: logs/predictions_map_2024_v24_final.png.

###Notes###
•	Sample Data: The data/ directory contains small subsets (e.g., 100 training points, 256x256 pixel rasters, sample shapefile) for testing. Full datasets are required for training and prediction.
•	CRS Compatibility: The code assumes coordinates in EPSG:4326. Ensure your data matches this CRS or adjust the LandUseDataset and LandUsePredictionDataset classes.
•	Performance: The scripts use a GPU if available (CUDA); otherwise, they fall back to CPU. Training and prediction with full datasets may require significant memory (e.g., 16GB RAM, 4GB VRAM).
•	Large Files: Full TIFF and shapefiles are not included due to size constraints. Instructions for obtaining them are provided above.
•	Customization: Modify the config dictionaries in src/land_use_classifier.py and src/land_use_predictor.py to adjust hyperparameters (e.g., patch_size, batch_size, grid_step).
•	Random Forest: The framework currently uses only deep learning. If random forest integration is planned, update the repository with relevant scripts and documentation.

###Computer Code Availability###
The source code, quick-test script, and sample data are available at https://github.com/JaneGondwegithub/LULC_classification_and_prediction. The repository is public and allows anonymous access.
License
Copyright (c) 2025 Jane Ferah Gondwe.
This project is licensed under the MIT License - see the LICENSE file for details.


