import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from scipy.ndimage import convolve
from imblearn.over_sampling import SMOTE
from scipy.ndimage import distance_transform_edt
import os
import logging
from datetime import datetime
import seaborn as sns
from rasterio.transform import rowcol

# Define CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class SpatialDropout(nn.Module):
    def __init__(self, drop_prob):
        super(SpatialDropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        batch, channels, height, width = x.size()
        mask = torch.rand(batch, channels, 1, 1, device=x.device) > self.drop_prob
        mask = mask.expand_as(x)
        return x * mask / (1 - self.drop_prob)

# Configuration
config = {
    "patch_size": 32,  # Window size, so patch is 64x64
    "batch_size": 32,
    "num_classes": 5,
    "num_channels": 24,  # 5 spectral + 8 indices + 4 GLCM + 1 variance + 1 LBP + 1 water + 1 roads + 4 DEM
    "class1_threshold": 0.20,
    "class3_threshold": 0.15,
    "initial_lr": 0.0005,
    "num_epochs": 50,
    "patience": 10,
    "log_dir": r"C:\Users\janeg\Desktop\2024Pridictions\logs",
    "model_path": r"C:\Users\janeg\Desktop\2024Pridictions\logs\best_land_use_classifier_v27.pth",
    "validation_csv_path": r"C:\Users\janeg\Desktop\2024Pridictions\logs\validation_points_v27.csv",
}

# Setup logging with DEBUG level
os.makedirs(config["log_dir"], exist_ok=True)
log_file = os.path.join(config["log_dir"], f"training_log_v27_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])
logging.info(f"Training started with config: {config}")

# Paths
image_path = r"C:\Users\janeg\Desktop\2024Pridictions\2024_landsat_malawi\Landsat8_2024_Malawi_Merged_FixedNoData.tif"
dem_path = r"C:\Users\janeg\Desktop\2024Pridictions\Merge raster\Raster.tif"
roads_path = r"C:\Users\janeg\Desktop\2024Pridictions\roads1.tif"
ground_truth_csv = r"C:\Users\janeg\Desktop\2024Pridictions\Training_Points_Updated_2024.csv"

# Split ground truth into train and validation
gdf = pd.read_csv(ground_truth_csv)
train_gdf, val_gdf = train_test_split(gdf, test_size=0.2, stratify=gdf['Class'], random_state=42)
train_csv = os.path.join(config["log_dir"], "train_points.csv")
val_csv = os.path.join(config["log_dir"], "val_points.csv")
train_gdf.to_csv(train_csv, index=False)
val_gdf.to_csv(val_csv, index=False)
logging.info(f"Split ground truth: {len(train_gdf)} training points, {len(val_gdf)} validation points")
logging.info(f"Training class distribution:\n{train_gdf['Class'].value_counts()}")
logging.info(f"Validation class distribution:\n{val_gdf['Class'].value_counts()}")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Running on {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Dataset for training
class LandUseDataset(Dataset):
    def __init__(self, image_path, dem_path, roads_path, ground_truth_csv, window_size=32, is_train=True):
        self.image_path = image_path
        self.dem_path = dem_path
        self.roads_path = roads_path
        self.window_size = window_size
        self.is_train = is_train

        # Load metadata and verify bands
        with rasterio.open(self.image_path) as src:
            self.image_meta = src.meta.copy()
            self.width = src.width
            self.height = src.height
            self.transform = src.transform
            self.crs = src.crs
            self.num_bands = src.count
            self.band_descriptions = src.descriptions
            logging.info(f"Image raster has {self.num_bands} bands")
            logging.info(f"Band descriptions: {self.band_descriptions}")
            if self.num_bands < 5:
                raise ValueError(f"Expected at least 5 bands, got {self.num_bands}")

        with rasterio.open(self.dem_path) as src:
            self.dem_meta = src.meta.copy()
            self.dem_transform = src.transform
            self.dem_crs = src.crs
            self.resolution = src.res[0]

        with rasterio.open(self.roads_path) as src:
            self.roads_meta = src.meta.copy()
            self.roads_transform = src.transform
            self.roads_crs = src.crs

        # Load ground truth points
        gdf = pd.read_csv(ground_truth_csv)
        if 'Longitude' in gdf.columns and 'longitude' in gdf.columns:
            gdf['longitude'] = gdf['Longitude']
        elif 'Longitude' in gdf.columns:
            gdf['longitude'] = gdf['Longitude']
        self.points = gdf[['longitude', 'latitude', 'Class']].dropna()

        # Convert points to pixel coordinates
        self.positions = []
        for idx, row in self.points.iterrows():
            lon, lat = row['longitude'], row['latitude']
            try:
                row_idx, col_idx = rowcol(self.transform, lon, lat)
                if 0 <= row_idx < self.height and 0 <= col_idx < self.width:
                    self.positions.append((row_idx, col_idx, int(row['Class'])))
                else:
                    logging.warning(f"Point at ({lon}, {lat}) is outside raster bounds")
            except Exception as e:
                logging.warning(f"Failed to convert coordinates ({lon}, {lat}) to pixel: {str(e)}")
        logging.info(f"Loaded {len(self.positions)} {'training' if is_train else 'validation'} points")

        # Initialize scalers
        self.scalers = [StandardScaler() for _ in range(config["num_channels"])]

    def fit_scalers(self):
        sample_patches = []
        sample_size = min(1000, len(self.positions))
        sample_indices = np.random.choice(len(self.positions), sample_size, replace=False)
        for idx in sample_indices:
            patch = self._get_patch(idx)
            if patch is not None and patch.shape[0] == config["num_channels"]:
                sample_patches.append(patch)
            else:
                logging.debug(f"Skipping invalid patch at index {idx}: shape {patch.shape if patch is not None else 'None'}")
        if len(sample_patches) < 10:  # Ensure enough valid patches
            raise ValueError(f"Too few valid patches ({len(sample_patches)}) for scaler fitting")
        sample_patches = np.array(sample_patches)
        logging.info(f"Fitting scalers on {len(sample_patches)} patches with shape {sample_patches.shape}")
        for channel in range(sample_patches.shape[1]):
            channel_data = sample_patches[:, channel, :, :].reshape(-1, 1)
            if np.any(np.isnan(channel_data)) or np.any(np.isinf(channel_data)):
                logging.warning(f"Channel {channel} contains NaN/Inf, replacing with zeros")
                channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=0.0, neginf=0.0)
            self.scalers[channel].fit(channel_data)
        logging.info("Scalers fitted on sample patches")

    def normalize(self, image):
        if image.shape[0] != config["num_channels"]:
            logging.error(f"Invalid image shape {image.shape}, expected {config['num_channels']} channels")
            return image
        for channel in range(image.shape[0]):
            channel_data = image[channel, :, :].reshape(-1, 1)
            if np.any(np.isnan(channel_data)) or np.any(np.isinf(channel_data)):
                logging.warning(f"Channel {channel} contains NaN/Inf during normalization, replacing with zeros")
                channel_data = np.nan_to_num(channel_data, nan=0.0, posinf=0.0, neginf=0.0)
            image[channel, :, :] = self.scalers[channel].transform(channel_data).reshape(image.shape[1], image.shape[2])
        return image

    def _get_patch(self, idx):
        row, col, _ = self.positions[idx]
        window = Window(col - self.window_size, row - self.window_size, self.window_size * 2, self.window_size * 2)

        # Read Landsat-8 patch (Blue, Green, Red, NIR, SWIR1)
        try:
            with rasterio.open(self.image_path) as src:
                image = src.read([2, 3, 4, 5, 6], window=window)  # Bands 2-6
            logging.debug(f"Read image patch at ({row}, {col}): shape {image.shape}")
        except Exception as e:
            logging.warning(f"Failed to read image patch at position ({row}, {col}): {str(e)}")
            return None

        if image.shape[0] != 5:
            logging.warning(f"Expected 5 bands, got {image.shape[0]} at ({row}, {col})")
            return None

        if image.shape[1] != self.window_size * 2 or image.shape[2] != self.window_size * 2:
            padded_image = np.zeros((5, self.window_size * 2, self.window_size * 2), dtype=image.dtype)
            h, w = image.shape[1], image.shape[2]
            padded_image[:, :h, :w] = image
            image = padded_image

        # Check for NaN/Inf in raw bands
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            logging.warning(f"Raw image patch at ({row}, {col}) contains NaN/Inf")
            return None

        # Read DEM patch
        dem_window = Window(col - self.window_size, row - self.window_size, self.window_size * 2, self.window_size * 2)
        try:
            with rasterio.open(self.dem_path) as src:
                dem = src.read(1, window=dem_window)
        except Exception as e:
            logging.warning(f"Failed to read DEM patch at position ({row}, {col}): {str(e)}")
            return None

        if dem.shape != (self.window_size * 2, self.window_size * 2):
            padded_dem = np.zeros((self.window_size * 2, self.window_size * 2), dtype=dem.dtype)
            h, w = dem.shape[0], dem.shape[1]
            padded_dem[:h, :w] = dem
            dem = padded_dem

        if np.any(np.isnan(dem)) or np.any(np.isinf(dem)):
            logging.warning(f"DEM patch at ({row}, {col}) contains NaN/Inf")
            return None

        # Read roads patch
        roads_window = Window(col - self.window_size, row - self.window_size, self.window_size * 2, self.window_size * 2)
        try:
            with rasterio.open(self.roads_path) as src:
                roads = src.read(1, window=roads_window)
        except Exception as e:
            logging.warning(f"Failed to read roads patch at position ({row}, {col}): {str(e)}")
            return None

        if roads.shape != (self.window_size * 2, self.window_size * 2):
            padded_roads = np.zeros((self.window_size * 2, self.window_size * 2), dtype=roads.dtype)
            h, w = roads.shape[0], roads.shape[1]
            padded_roads[:h, :w] = roads
            roads = padded_roads

        # Compute features
        blue = image[0].astype(np.float32)
        green = image[1].astype(np.float32)
        red = image[2].astype(np.float32)
        nir = image[3].astype(np.float32)
        swir1 = image[4].astype(np.float32)

        # Compute indices
        ndvi = (nir - red) / (nir + red + 1e-8)
        ndvi = np.clip(ndvi, -1, 1)
        logging.debug(f"NDVI shape at ({row}, {col}): {ndvi.shape}")

        ndwi = (green - nir) / (green + nir + 1e-8)
        ndwi = np.clip(ndwi, -1, 1)
        logging.debug(f"NDWI shape at ({row}, {col}): {ndwi.shape}")

        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8)
        evi = np.clip(evi, -1.5, 1.5)
        logging.debug(f"EVI shape at ({row}, {col}): {evi.shape}")

        L = 0.5
        savi = ((nir - red) * (1 + L)) / (nir + red + L + 1e-8)
        savi = np.clip(savi, -1, 1)
        logging.debug(f"SAVI shape at ({row}, {col}): {savi.shape}")

        msavi = (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / (2 + 1e-8)
        msavi = np.clip(msavi, -1, 1)
        logging.debug(f"MSAVI shape at ({row}, {col}): {msavi.shape}")

        ndbi = (nir - red) / (nir + red + 1e-8)
        ndbi = np.clip(ndbi, 0, 1)
        logging.debug(f"NDBI shape at ({row}, {col}): {ndbi.shape}")

        ndmi = (nir - swir1) / (nir + swir1 + 1e-8)
        ndmi = np.clip(ndmi, -1, 1)
        logging.debug(f"NDMI shape at ({row}, {col}): {ndmi.shape}")

        bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + 1e-8)
        bsi = np.clip(bsi, -1, 1)
        logging.debug(f"BSI shape at ({row}, {col}): {bsi.shape}")

        # Check indices for NaN/Inf
        indices = [ndvi, ndwi, evi, savi, msavi, ndbi, ndmi, bsi]
        for i, idx_data in enumerate(indices):
            if np.any(np.isnan(idx_data)) or np.any(np.isinf(idx_data)):
                logging.warning(f"Index {i} (e.g., {'NDVI' if i==0 else 'NDWI' if i==1 else 'EVI' if i==2 else 'SAVI' if i==3 else 'MSAVI' if i==4 else 'NDBI' if i==5 else 'NDMI' if i==6 else 'BSI'}) at ({row}, {col}) contains NaN/Inf")
                return None

        # GLCM features
        nir_min, nir_max = nir.min(), nir.max()
        if nir_max == nir_min or np.isnan(nir_max) or np.isnan(nir_min):
            logging.warning(f"Invalid NIR range at ({row}, {col}): min={nir_min}, max={nir_max}")
            return None
        nir_uint8 = (nir - nir_min) / (nir_max - nir_min + 1e-8) * 255
        nir_uint8 = np.nan_to_num(nir_uint8, nan=0.0, posinf=0.0, neginf=0.0).astype(np.uint8)
        glcm = graycomatrix(nir_uint8, distances=[1], angles=[0, 45, 90, 135], levels=256, symmetric=True, normed=True)
        contrast = np.mean([graycoprops(glcm, 'contrast')[0, i] for i in range(4)])
        correlation = np.mean([graycoprops(glcm, 'correlation')[0, i] for i in range(4)])
        homogeneity = np.mean([graycoprops(glcm, 'homogeneity')[0, i] for i in range(4)])
        energy = np.mean([graycoprops(glcm, 'energy')[0, i] for i in range(4)])
        contrast_map = np.full((self.window_size * 2, self.window_size * 2), contrast, dtype=np.float32)
        correlation_map = np.full((self.window_size * 2, self.window_size * 2), correlation, dtype=np.float32)
        homogeneity_map = np.full((self.window_size * 2, self.window_size * 2), homogeneity, dtype=np.float32)
        energy_map = np.full((self.window_size * 2, self.window_size * 2), energy, dtype=np.float32)
        logging.debug(f"GLCM shapes at ({row}, {col}): contrast={contrast_map.shape}, correlation={correlation_map.shape}, homogeneity={homogeneity_map.shape}, energy={energy_map.shape}")

        # NIR variance
        nir_variance = np.var(nir)
        if np.isnan(nir_variance) or np.isinf(nir_variance):
            logging.warning(f"NIR variance at ({row}, {col}) is invalid: {nir_variance}")
            return None
        variance_map = np.full((self.window_size * 2, self.window_size * 2), nir_variance, dtype=np.float32)
        logging.debug(f"NIR variance shape at ({row}, {col}): {variance_map.shape}")

        # LBP
        lbp = local_binary_pattern(nir_uint8, P=8, R=1, method='uniform')
        lbp = lbp.astype(np.float32)
        lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-8)
        if np.any(np.isnan(lbp)) or np.any(np.isinf(lbp)):
            logging.warning(f"LBP at ({row}, {col}) contains NaN/Inf")
            return None
        logging.debug(f"LBP shape at ({row}, {col}): {lbp.shape}")

        # Proximity-to-water
        water_mask = (ndwi > 0.2) & (nir < 0.1)
        proximity_to_water = distance_transform_edt(1 - water_mask)
        proximity_to_water = proximity_to_water / (proximity_to_water.max() + 1e-8)
        if np.any(np.isnan(proximity_to_water)) or np.any(np.isinf(proximity_to_water)):
            logging.warning(f"Proximity-to-water at ({row}, {col}) contains NaN/Inf")
            return None
        logging.debug(f"Proximity-to-water shape at ({row}, {col}): {proximity_to_water.shape}")

        # Proximity-to-roads
        road_mask = roads > 0
        proximity_to_roads = distance_transform_edt(1 - road_mask)
        proximity_to_roads = proximity_to_roads / (proximity_to_roads.max() + 1e-8)
        if np.any(np.isnan(proximity_to_roads)) or np.any(np.isinf(proximity_to_roads)):
            logging.warning(f"Proximity-to-roads at ({row}, {col}) contains NaN/Inf")
            return None
        logging.debug(f"Proximity-to-roads shape at ({row}, {col}): {proximity_to_roads.shape}")

        # DEM-derived features
        dem = dem.astype(np.float32)
        dy, dx = np.gradient(dem, self.resolution)
        slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)
        slope = slope.astype(np.float32)
        if np.any(np.isnan(slope)) or np.any(np.isinf(slope)):
            logging.warning(f"Slope at ({row}, {col}) contains NaN/Inf")
            return None
        logging.debug(f"Slope shape at ({row}, {col}): {slope.shape}")

        aspect = np.arctan2(dy, dx) * (180 / np.pi)
        aspect = np.where(aspect < 0, aspect + 360, aspect)
        aspect = aspect.astype(np.float32)
        if np.any(np.isnan(aspect)) or np.any(np.isinf(aspect)):
            logging.warning(f"Aspect at ({row}, {col}) contains NaN/Inf")
            return None
        logging.debug(f"Aspect shape at ({row}, {col}): {aspect.shape}")

        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        try:
            tri = np.abs(convolve(dem, kernel, mode='constant', cval=0.0))
            tri = tri.astype(np.float32)
            if np.any(np.isnan(tri)) or np.any(np.isinf(tri)):
                logging.warning(f"TRI at ({row}, {col}) contains NaN/Inf")
                return None
            logging.debug(f"TRI shape at ({row}, {col}): {tri.shape}")
        except Exception as e:
            logging.warning(f"TRI computation failed at position ({row}, {col}): {str(e)}")
            tri = np.zeros_like(dem)

        # Stack features (24 channels)
        features = [
            ('Blue', image[0:1, :, :]),  # Band 2
            ('Green', image[1:2, :, :]),  # Band 3
            ('Red', image[2:3, :, :]),  # Band 4
            ('NIR', image[3:4, :, :]),  # Band 5
            ('SWIR1', image[4:5, :, :]),  # Band 6
            ('NDVI', ndvi[np.newaxis, :, :]),
            ('NDWI', ndwi[np.newaxis, :, :]),
            ('EVI', evi[np.newaxis, :, :]),
            ('SAVI', savi[np.newaxis, :, :]),
            ('MSAVI', msavi[np.newaxis, :, :]),
            ('NDBI', ndbi[np.newaxis, :, :]),
            ('NDMI', ndmi[np.newaxis, :, :]),
            ('BSI', bsi[np.newaxis, :, :]),
            ('GLCM_Contrast', contrast_map[np.newaxis, :, :]),
            ('GLCM_Correlation', correlation_map[np.newaxis, :, :]),
            ('GLCM_Homogeneity', homogeneity_map[np.newaxis, :, :]),
            ('GLCM_Energy', energy_map[np.newaxis, :, :]),
            ('NIR_Variance', variance_map[np.newaxis, :, :]),
            ('LBP', lbp[np.newaxis, :, :]),
            ('Proximity_to_Water', proximity_to_water[np.newaxis, :, :]),
            ('Proximity_to_Roads', proximity_to_roads[np.newaxis, :, :]),
            ('DEM', dem[np.newaxis, :, :]),
            ('Slope', slope[np.newaxis, :, :]),
            ('Aspect', aspect[np.newaxis, :, :]),
            ('TRI', tri[np.newaxis, :, :]),
        ]

        # Log feature shapes
        logging.debug(f"Feature shapes at ({row}, {col}):")
        for name, feat in features:
            logging.debug(f"  {name}: {feat.shape}")

        # Concatenate features
        image = np.concatenate([feat for _, feat in features], axis=0)

        if image.shape[0] != config["num_channels"]:
            logging.error(f"Final patch at ({row}, {col}) has incorrect channels: {image.shape[0]}, expected {config['num_channels']}")
            logging.error(f"Feature count: {len(features)}")
            return None

        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            logging.warning(f"Final patch data at ({row}, {col}) contains NaN/Inf")
            return None

        logging.debug(f"Final patch at ({row}, {col}): shape {image.shape}")
        return image

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        patch = self._get_patch(idx)
        if patch is None:
            return torch.zeros((config["num_channels"], self.window_size * 2, self.window_size * 2), dtype=torch.float32), 0
        patch = self.normalize(patch)
        return torch.tensor(patch, dtype=torch.float32), self.positions[idx][2]

# Load and preprocess data
dataset = LandUseDataset(image_path, dem_path, roads_path, train_csv, window_size=config["patch_size"])
dataset.fit_scalers()

# Apply SMOTE to balance classes
features = []
labels = []
for idx in range(len(dataset)):
    patch, label = dataset[idx]
    features.append(patch.numpy().flatten())
    labels.append(label)
features = np.array(features)
labels = np.array(labels)

# Enhanced SMOTE oversampling based on class distribution
max_count = np.bincount(labels).max()  # 599 (Class 1)
sampling_strategy = {
    0: int(max_count * 1.5),  # Water: 290 -> ~898
    1: max_count,            # Built-up: 599 -> 599
    2: int(max_count * 2.0),  # Bareland: 257 -> ~1198
    3: max_count,            # Agriculture: 560 -> 599
    4: int(max_count * 2.0)   # Forest: 166 -> ~1198
}
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
features_resampled, labels_resampled = smote.fit_resample(features, labels)

# Create a new dataset with resampled data
class ResampledDataset(Dataset):
    def __init__(self, features, labels, num_channels, patch_size):
        self.features = features
        self.labels = labels
        self.num_channels = num_channels
        self.patch_size = patch_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx].reshape(self.num_channels, self.patch_size * 2, self.patch_size * 2)
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), label

resampled_dataset = ResampledDataset(features_resampled, labels_resampled, config["num_channels"], config["patch_size"])
train_loader = DataLoader(resampled_dataset, batch_size=config["batch_size"], shuffle=True)

# Define the model
model = models.resnet18()
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = True
model.conv1 = nn.Conv2d(config["num_channels"], 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model_spatial_dropout = nn.Sequential(
    model.conv1,
    model.bn1,
    model.relu,
    SpatialDropout(drop_prob=0.2)
)
model.conv1 = model_spatial_dropout
model = nn.Sequential(
    *list(model.children())[:-2],
    CBAM(in_planes=512),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Dropout(0.5),
    nn.BatchNorm1d(512),
    nn.Linear(512, config["num_classes"])
)
model = model.to(device)

# Training setup
class_counts = np.bincount(labels)
class_weights = 1.0 / (class_counts + 1e-8)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=config["initial_lr"], weight_decay=5e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Training loop
best_val_loss = float('inf')
patience_counter = 0
for epoch in range(config["num_epochs"]):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    logging.info(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {epoch_loss:.4f}")

    # Validation with accuracy and confusion matrix
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        val_dataset = LandUseDataset(image_path, dem_path, roads_path, val_csv, window_size=config["patch_size"], is_train=False)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    per_class_acc = conf_matrix.diagonal() / (conf_matrix.sum(axis=1) + 1e-8)

    # Log validation metrics
    logging.info(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
    logging.info(f"Confusion Matrix:\n{conf_matrix}")
    for i, acc in enumerate(per_class_acc):
        logging.info(f"Class {i} ({['Water', 'Built-up', 'Bareland', 'Agriculture', 'Forest'][i]}) accuracy: {acc:.4f}")

    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - Epoch {epoch+1}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plot_path = os.path.join(config['log_dir'], f'conf_matrix_epoch_{epoch+1}.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Confusion matrix plot saved to {plot_path}")

    scheduler.step(val_loss)
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), config["model_path"])
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= config["patience"]:
            logging.info("Early stopping triggered")
            break

logging.info("Training completed")

# Save validation points
val_points = pd.DataFrame({
    'longitude': val_dataset.points['longitude'],
    'latitude': val_dataset.points['latitude'],
    'Class': val_dataset.points['Class'],
    'Is_Synthetic': [0] * len(val_dataset.points)
})
val_points.to_csv(config["validation_csv_path"], index=False)
logging.info(f"Validation points saved to {config['validation_csv_path']}")