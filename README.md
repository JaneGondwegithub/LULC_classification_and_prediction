#Land Use Classifier#
##Purpose##
This repository contains a Python script for land use classification using Landsat-8 imagery and Digital Elevation Model (DEM) data. The model is built with PyTorch, leveraging a ResNet-18 backbone, Focal Loss, SMOTE for class balancing, and spatial dropout for regularization. It classifies land use types based on geospatial training points, incorporating spectral indices (e.g., NDVI, NDWI) and terrain features (e.g., slope, aspect).

###Repository Structure###
•	src/land_use_classifier.py: Main script for training the land use classification model.
•	tests/quick_test.py: Quick-test script to verify dataset loading and patch extraction.
•	data/sample_training_points.csv: Sample training points (100 points, subset of full dataset).
•	data/sample_landsat.tif: Sample Landsat-8 image (cropped to 256x256 pixels).
•	data/sample_dem.tif: Sample DEM (cropped to 256x256 pixels).
•	requirements.txt: Python dependencies.
•	LICENSE: MIT License.

###Setup###
1.	Clone the Repository:
2.	git clone https://github.com/JaneGondwegithub/LULC_classification_and_prediction.git 
3.	cd land-use-classifier
4.	Install Dependencies:
5.	pip install -r requirements.txt
Ensure you have PyTorch with CUDA support for GPU usage (see PyTorch installation guide).
6.	Python Version: Use Python 3.8 or higher.
7.	Environment Note: Some dependencies (e.g., rasterio, geopandas) may require additional system libraries (e.g., GDAL). See platform-specific installation guides if needed.
   
###Quick-Test Instructions###
To verify the code works with the provided sample data:
1.	Run the test script:
2.	python tests/quick_test.py
3.	Expected Output: Prints the shape and label of the first 5 valid patches from the sample dataset (e.g., Sample 0: Label=2, Patch shape=torch.Size([17, 64, 64])).
4.	The test uses data/sample_training_points.csv, data/sample_landsat.tif, and data/sample_dem.tif to load and process a few patches.
   
###Running the Full Model###
To train the model with the full datasets:
1.	Obtain Full Datasets:
o	Landsat-8 Image: Download Landsat8_2020_Malawi_Merged_FixedNoData.tif from USGS EarthExplorer or your data source. Ensure it has 4 bands and no excessive NoData values.
o	DEM: Download Raster.tif from SRTM or your data provider. Ensure it aligns spatially with the Landsat-8 image.
o	Training Points: Use the full Cleaned_Training_Points_2020.csv, which should have columns latitude, longitude, and Class (integers 0-4). The structure matches data/sample_training_points.csv.
o	Place these files in the data/ directory or update the paths in src/land_use_classifier.py (e.g., modify train_csv_path, image_path, dem_path).
2.	Run the Main Script:
3.	python src/land_use_classifier.py
4.	Outputs:
o	Model weights saved to logs/best_land_use_forecaster_v24_final.pth.
o	Training logs saved to logs/forecast_training_log_v24_*.log.
o	Plots (loss, accuracy, misclassification map) saved to logs/.
o	Validation points with predictions saved to logs/validation_points_v24_final.csv.

###Notes###
•	Sample Data: The provided sample data (data/) is a small subset for testing purposes. It includes a cropped Landsat-8 image, DEM, and 100 training points. The full datasets are required for training the complete model.
•	CRS Compatibility: The code assumes coordinates in EPSG:4326. Ensure your data matches this CRS or adjust the LandUseDataset class to handle transformations.
•	Performance: The script uses a GPU if available (CUDA); otherwise, it falls back to CPU. Training with the full dataset may require significant memory (e.g., 16GB RAM, 4GB VRAM).
•	Large Files: The full TIFF files are not included due to size constraints. Instructions for obtaining them are provided above.
•	Customization: Modify the config dictionary in src/land_use_classifier.py to adjust hyperparameters (e.g., epochs, batch_size, patch_size).

###Computer Code Availability###
The source code, quick-test script, and sample data are available at https://github.com/JaneGondwegithub/LULC_classification_and_prediction.git. The repository is public and allows anonymous access.

###License###
Copyright (c) 2025 Jane Ferah Gondwe.

