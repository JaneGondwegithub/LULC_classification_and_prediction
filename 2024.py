import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import models
import torchvision.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops
from imblearn.over_sampling import SMOTE
import rasterio
from pyproj import Transformer
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import logging
from datetime import datetime
import psutil
import imblearn
from scipy.ndimage import convolve

# Silence joblib warning
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Log imblearn version
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info(f"imblearn version: {imblearn.__version__}")

# Define Focal Loss with Label Smoothing
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.5, alpha=None, reduction='mean', label_smoothing=0.05):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        smoothed_labels = torch.full_like(inputs, self.label_smoothing / (num_classes - 1))
        smoothed_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        
        logpt = torch.log_softmax(inputs, dim=1)
        ce_loss = -torch.sum(smoothed_labels * logpt, dim=1)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha[targets] * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# Define Spatial Dropout for Regularization
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

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
config = {
    "patch_size": 32,
    "epochs": 18,  # Reduced to stop at Epoch 18
    "batch_size": 16,
    "num_classes": 5,
    "log_dir": r"C:\Users\janeg\Desktop\2024Pridictions\logs",
    "initial_lr": 1e-4,
    "weight_decay": 1e-3,
    "dropout_rate": 0.85,
    "patience": 30,
    "num_augmentations": 2,
    "num_channels": 17,
}
print("Config loaded!")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {'GPU' if torch.cuda.is_available() else 'CPU'}")

# Logging Setup
os.makedirs(config["log_dir"], exist_ok=True)
log_file = os.path.join(config["log_dir"], f"forecast_training_log_v24_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.getLogger().addHandler(logging.FileHandler(log_file))
logging.info(f"Training started with config: {config}")
print(f"Logging started at: {log_file}")

# Paths
train_csv_path = r"C:\Users\janeg\Desktop\2024Pridictions\Cleaned_Training_Points_2020.csv"
image_path = r"C:\Users\janeg\Desktop\2024Pridictions\2020_landsat_malawi\Landsat8_2020_Malawi_Merged_FixedNoData.tif"
dem_path = r"C:\Users\janeg\Desktop\2024Pridictions\Merge raster\Raster.tif"
model_save_path = os.path.join(config["log_dir"], "best_land_use_forecaster_v24_final.pth")

# Step 1: Verify the Landsat-8 Image
def verify_image(image_path):
    try:
        with rasterio.open(image_path) as src:
            print(f"Verifying Landsat-8 image at {image_path}")
            print("Number of bands:", src.count)
            print("Shape:", src.shape)
            print("CRS:", src.crs)
            print("Transform:", src.transform)
            print("Bounds:", src.bounds)
            print("NoData value:", src.nodata)
            data = src.read(masked=True)
            print("Data min/max (per band, ignoring NoData):")
            for i in range(src.count):
                band_data = data[i]
                valid_data = band_data[~band_data.mask]
                if valid_data.size > 0:
                    print(f"Band {i+1}: min={valid_data.min()}, max={valid_data.max()}")
                else:
                    print(f"Band {i+1}: No valid data (all pixels are NoData)")
                    raise ValueError(f"Band {i+1} contains no valid data. Check the image for excessive NoData values.")
            return src.bounds
    except Exception as e:
        logging.error(f"Failed to verify Landsat-8 image: {str(e)}")
        raise

# Verify the image
image_bounds = verify_image(image_path)

# Step 2: Verify the Training Points CSV
def verify_training_points(csv_path, image_bounds):
    df = pd.read_csv(csv_path)
    print(f"Loaded training points from {csv_path}")
    print(f"Total points: {len(df)}")
    
    required_columns = ['latitude', 'Longitude', 'Class']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in CSV: {missing_columns}")
    
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude'])
    print(f"Points after dropping NaN coordinates: {len(df)}")
    
    df = df[
        (df['longitude'] >= image_bounds.left) & (df['longitude'] <= image_bounds.right) &
        (df['latitude'] >= image_bounds.bottom) & (df['latitude'] <= image_bounds.top)
    ]
    print(f"Points within image bounds: {len(df)}")
    
    df['Class'] = df['Class'].astype(int)
    unique_classes = df['Class'].unique()
    print(f"Unique classes: {sorted(unique_classes)}")
    if not all(cls in range(config["num_classes"]) for cls in unique_classes):
        raise ValueError(f"Classes must be in range 0 to {config['num_classes']-1}. Found: {unique_classes}")
    
    return df

# Verify training points
train_df = verify_training_points(train_csv_path, image_bounds)

# Step 3: Define the Dataset with DEM and Derived Features
class LandUseDataset(Dataset):
    def __init__(self, csv_file, image_path, dem_path, window_size=32):
        self.data = pd.read_csv(csv_file)
        self.image_path = image_path
        self.dem_path = dem_path
        self.window_size = window_size

        # Load Landsat-8 metadata
        with rasterio.open(self.image_path) as src:
            self.transform_raster = src.transform
            self.crs = src.crs
            self.num_bands = src.count
            self.image_height = src.height
            self.image_width = src.width
        logging.info(f"Landsat-8 image has {self.num_bands} bands")

        # Load DEM metadata
        try:
            with rasterio.open(self.dem_path) as src:
                self.transform_dem = src.transform
                self.crs_dem = src.crs
                self.num_bands_dem = src.count
                self.bounds_dem = src.bounds
                self.resolution = src.res[0]
                self.dem_height = src.height
                self.dem_width = src.width
            logging.info(f"DEM has {self.num_bands_dem} bands, CRS: {self.crs_dem}, Bounds: {self.bounds_dem}, Resolution: {self.resolution}m")
        except Exception as e:
            logging.error(f"Failed to load DEM: {str(e)}")
            raise ValueError(f"Cannot open DEM file at {self.dem_path}")

        # Coordinate transformers
        if self.crs != 'EPSG:4326':
            self.transformer_image = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        else:
            self.transformer_image = None
        if self.crs_dem != 'EPSG:4326':
            self.transformer_dem = Transformer.from_crs("EPSG:4326", self.crs_dem, always_xy=True)
        else:
            self.transformer_dem = None

        self.data['latitude'] = pd.to_numeric(self.data['latitude'], errors='coerce')
        self.data['longitude'] = pd.to_numeric(self.data['Longitude'], errors='coerce')
        self.data = self.data.dropna(subset=['latitude', 'longitude'])
        self.data = self.data[
            (self.data['latitude'].between(-90, 90)) & 
            (self.data['longitude'].between(-180, 180))
        ]
        logging.info(f"Filtered dataset size: {len(self.data)}")
        logging.info(f"Class distribution: {self.data['Class'].value_counts().sort_index().to_dict()}")

        # Initialize scalers for each channel
        self.scalers = [StandardScaler() for _ in range(config["num_channels"])]

    def fit_scalers(self, X_patches):
        for channel in range(X_patches.shape[1]):
            channel_data = X_patches[:, channel, :, :].reshape(-1, 1)
            self.scalers[channel].fit(channel_data)

    def normalize(self, image):
        for channel in range(image.shape[0]):
            channel_data = image[channel, :, :].reshape(-1, 1)
            image[channel, :, :] = self.scalers[channel].transform(channel_data).reshape(image.shape[1], image.shape[2])
        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        lat = self.data.iloc[idx]['latitude']
        lon = self.data.iloc[idx]['longitude']
        label = self.data.iloc[idx]['Class']

        # Transform coordinates for Landsat-8
        if self.transformer_image:
            lon_image, lat_image = self.transformer_image.transform(lon, lat)
        else:
            lon_image, lat_image = lon, lat
        col, row = ~self.transform_raster * (lon_image, lat_image)

        # Transform coordinates for DEM
        if self.transformer_dem:
            lon_dem, lat_dem = self.transformer_dem.transform(lon, lat)
        else:
            lon_dem, lat_dem = lon, lat
        col_dem, row_dem = ~self.transform_dem * (lon_dem, lat_dem)

        if np.isnan(row) or np.isnan(col) or np.isnan(row_dem) or np.isnan(col_dem):
            if idx % 100 == 0:
                logging.warning(f"Invalid patch at index {idx}: lat={lat}, lon={lon}")
            return torch.zeros((config["num_channels"], self.window_size * 2, self.window_size * 2), dtype=torch.float32), -1

        row, col = int(row), int(col)
        row_dem, col_dem = int(row_dem), int(col_dem)

        # Check if Landsat-8 coordinates are within bounds
        if not (0 <= row - self.window_size < self.image_height and 0 <= col - self.window_size < self.image_width):
            if idx % 100 == 0:
                logging.warning(f"Index {idx}: Landsat-8 coordinates out of bounds (row={row}, col={col})")
            return torch.zeros((config["num_channels"], self.window_size * 2, self.window_size * 2), dtype=torch.float32), -1

        # Check if DEM coordinates are within bounds
        if not (0 <= row_dem - self.window_size < self.dem_height and 0 <= col_dem - self.window_size < self.dem_width):
            if idx % 100 == 0:
                logging.warning(f"Index {idx}: DEM coordinates out of bounds (row_dem={row_dem}, col_dem={col_dem})")
            return torch.zeros((config["num_channels"], self.window_size * 2, self.window_size * 2), dtype=torch.float32), -1

        # Read Landsat-8 patch
        window = rasterio.windows.Window(
            col - self.window_size, row - self.window_size,
            self.window_size * 2, self.window_size * 2
        )
        try:
            with rasterio.open(self.image_path) as src:
                image = src.read(window=window, masked=True)
                image = image.filled(fill_value=0)
        except Exception as e:
            if idx % 100 == 0:
                logging.warning(f"Failed to read Landsat-8 patch at index {idx}: {str(e)}")
            return torch.zeros((config["num_channels"], self.window_size * 2, self.window_size * 2), dtype=torch.float32), -1

        if image.shape[1] != self.window_size * 2 or image.shape[2] != self.window_size * 2:
            padded_image = np.zeros((self.num_bands, self.window_size * 2, self.window_size * 2), dtype=image.dtype)
            h, w = image.shape[1], image.shape[2]
            padded_image[:, :h, :w] = image
            image = padded_image

        # Read DEM patch
        window_dem = rasterio.windows.Window(
            col_dem - self.window_size, row_dem - self.window_size,
            self.window_size * 2, self.window_size * 2
        )
        try:
            with rasterio.open(self.dem_path) as src:
                dem = src.read(1, window=window_dem, masked=True)
                dem = dem.filled(fill_value=0)
        except Exception as e:
            if idx % 100 == 0:
                logging.warning(f"Failed to read DEM patch at index {idx}: {str(e)}")
            return torch.zeros((config["num_channels"], self.window_size * 2, self.window_size * 2), dtype=torch.float32), -1

        if dem.shape != (self.window_size * 2, self.window_size * 2):
            padded_dem = np.zeros((self.window_size * 2, self.window_size * 2), dtype=dem.dtype)
            h, w = dem.shape[0], dem.shape[1]
            padded_dem[:h, :w] = dem
            dem = padded_dem

        if dem.mean() == 0:
            if idx % 100 == 0:
                logging.warning(f"Index {idx}: Invalid DEM data (mean=0)")
            return torch.zeros((config["num_channels"], self.window_size * 2, self.window_size * 2), dtype=torch.float32), -1

        # Compute NDVI, NDWI, EVI, SAVI, BSI
        blue = image[0].astype(np.float32)
        green = image[1].astype(np.float32)
        red = image[2].astype(np.float32)
        nir = image[3].astype(np.float32)

        ndvi = (nir - red) / (nir + red + 1e-8)
        ndvi = np.clip(ndvi, -1, 1)

        ndwi = (green - nir) / (green + nir + 1e-8)
        ndwi = np.clip(ndwi, -1, 1)

        evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1 + 1e-8)
        evi = np.clip(evi, -1.5, 1.5)

        L = 0.5
        savi = ((nir - red) * (1 + L)) / (nir + red + L + 1e-8)
        savi = np.clip(savi, -1, 1)

        bsi = ((red + nir) - (blue + green)) / ((red + nir) + (blue + green) + 1e-8)
        bsi = np.clip(bsi, -1, 1)

        # Compute GLCM features
        nir_uint8 = (nir - nir.min()) / (nir.max() - nir.min() + 1e-8) * 255
        nir_uint8 = nir_uint8.astype(np.uint8)
        glcm = graycomatrix(nir_uint8, distances=[1], angles=[0, 45, 90, 135], levels=256, symmetric=True, normed=True)
        contrast = np.mean([graycoprops(glcm, 'contrast')[0, i] for i in range(4)])
        correlation = np.mean([graycoprops(glcm, 'correlation')[0, i] for i in range(4)])
        homogeneity = np.mean([graycoprops(glcm, 'homogeneity')[0, i] for i in range(4)])
        energy = np.mean([graycoprops(glcm, 'energy')[0, i] for i in range(4)])
        contrast_map = np.full((self.window_size * 2, self.window_size * 2), contrast, dtype=np.float32)
        correlation_map = np.full((self.window_size * 2, self.window_size * 2), correlation, dtype=np.float32)
        homogeneity_map = np.full((self.window_size * 2, self.window_size * 2), homogeneity, dtype=np.float32)
        energy_map = np.full((self.window_size * 2, self.window_size * 2), energy, dtype=np.float32)

        # Compute slope (degrees)
        dem = dem.astype(np.float32)
        dy, dx = np.gradient(dem, self.resolution)
        slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)
        slope = slope.astype(np.float32)

        # Compute aspect (degrees, 0-360)
        aspect = np.arctan2(dy, dx) * (180 / np.pi)
        aspect = np.where(aspect < 0, aspect + 360, aspect)
        aspect = aspect.astype(np.float32)

        # Compute Terrain Ruggedness Index (TRI)
        kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
        try:
            tri = np.abs(convolve(dem, kernel, mode='constant', cval=0.0))
            tri = tri.astype(np.float32)
        except Exception as e:
            if idx % 100 == 0:
                logging.warning(f"TRI computation failed at index {idx}: {str(e)}")
            tri = np.zeros_like(dem)

        # Stack features (17 channels: 4 bands + 5 indices + 4 GLCM + elevation + slope + aspect + TRI)
        image = np.concatenate([
            image,
            ndvi[np.newaxis, :, :],
            ndwi[np.newaxis, :, :],
            evi[np.newaxis, :, :],
            savi[np.newaxis, :, :],
            bsi[np.newaxis, :, :],
            contrast_map[np.newaxis, :, :],
            correlation_map[np.newaxis, :, :],
            homogeneity_map[np.newaxis, :, :],
            energy_map[np.newaxis, :, :],
            dem[np.newaxis, :, :],
            slope[np.newaxis, :, :],
            aspect[np.newaxis, :, :],
            tri[np.newaxis, :, :]
        ], axis=0)

        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            if idx % 100 == 0:
                logging.warning(f"Invalid image data at index {idx}: NaN or Inf detected")
            return torch.zeros((config["num_channels"], self.window_size * 2, self.window_size * 2), dtype=torch.float32), -1

        return torch.tensor(image, dtype=torch.float32), label

# Step 4: Enhanced Data Augmentation with Custom Jittering
class CustomAugment:
    def __init__(self, num_augmentations=2):
        self.num_augmentations = num_augmentations
        self.gaussian_blur = T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))

    def adjust_brightness_contrast(self, img, brightness_factor, contrast_factor):
        mean = img.mean(dim=(1, 2), keepdim=True)
        img = (img - mean) * contrast_factor + mean + brightness_factor
        return img

    def __call__(self, img):
        augmented_images = [img]
        for _ in range(self.num_augmentations):
            x = img.clone()
            k = np.random.choice([1, 2, 3])
            x = torch.rot90(x, k, dims=(1, 2))
            if np.random.rand() > 0.5:
                x = torch.flip(x, dims=[1])
            if np.random.rand() > 0.5:
                x = torch.flip(x, dims=[2])
            shift_x, shift_y = np.random.randint(-2, 3, 2)
            x = torch.roll(x, shifts=(shift_x, shift_y), dims=(1, 2))
            spectral_channels = x[:8, :, :].clone()
            brightness_factor = torch.FloatTensor(8, 1, 1).uniform_(-0.2, 0.2).to(spectral_channels.device)
            contrast_factor = torch.FloatTensor(8, 1, 1).uniform_(0.8, 1.2).to(spectral_channels.device)
            spectral_channels = self.adjust_brightness_contrast(spectral_channels, brightness_factor, contrast_factor)
            x[:8, :, :] = spectral_channels
            if np.random.rand() > 0.5:
                x[:8, :, :] = self.gaussian_blur(x[:8, :, :])
            noise = torch.randn(x.size()) * 0.001
            x = x + noise
            x = torch.clamp(x, min=-3, max=3)
            augmented_images.append(x)
        return augmented_images

train_transforms = CustomAugment(num_augmentations=2)

# Step 5: Load the Dataset
train_dataset = LandUseDataset(
    csv_file=train_csv_path,
    image_path=image_path,
    dem_path=dem_path,
    window_size=config["patch_size"]
)

# Filter invalid samples
class FilteredDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.valid_indices = [i for i in range(len(dataset)) if dataset[i][1] != -1]
        logging.info(f"Valid samples after filtering: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        return self.dataset[self.valid_indices[idx]]

filtered_dataset = FilteredDataset(train_dataset)

# Step 6: Extract Patches and Apply Augmentation
X_patches, y_patches = [], []
original_indices = []

for idx in range(len(filtered_dataset)):
    patch, label = filtered_dataset[idx]
    if isinstance(patch, torch.Tensor):
        patch = patch.numpy()
    
    if patch.shape != (17, 64, 64):
        raise ValueError(f"Patch {idx} has unexpected shape: {patch.shape}")
    
    X_patches.append(patch)
    y_patches.append(label)
    original_indices.append(idx)
    
    augmented_patches = train_transforms(torch.from_numpy(patch))
    for aug_patch in augmented_patches:
        if aug_patch.shape != (17, 64, 64):
            aug_patch = aug_patch.permute(2, 0, 1) if aug_patch.shape == (64, 64, 17) else aug_patch
        X_patches.append(aug_patch.numpy())
        y_patches.append(label)
        original_indices.append(idx)

X_patches = np.array(X_patches)
y_patches = np.array(y_patches)
original_indices = np.array(original_indices)
logging.info(f"Post-augmentation: X shape={X_patches.shape}, Class distribution={np.unique(y_patches, return_counts=True)}")
print("Patches extracted and augmented!")

# Fit scalers before SMOTE
train_dataset.fit_scalers(X_patches)

# Normalize the patches
for idx in range(len(X_patches)):
    X_patches[idx] = train_dataset.normalize(X_patches[idx])

# Step 7: Fine-Tune SMOTE for Classes 1, 2, and 3
class_counts = dict(zip(*np.unique(y_patches, return_counts=True)))
max_count = max(class_counts.values())
sampling_strategy = {i: max_count for i in range(config["num_classes"])}
sampling_strategy[1] = int(max_count * 1.5)
sampling_strategy[2] = int(max_count * 1.7)
sampling_strategy[3] = int(max_count * 1.5)

smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=sampling_strategy)
logging.info(f"Pre-SMOTE counts: {dict(zip(*np.unique(y_patches, return_counts=True)))}")

# Log memory before SMOTE
process = psutil.Process()
logging.info(f"Memory before SMOTE: {process.memory_info().rss / 1024**2:.2f} MB")

X_patches_reshaped = X_patches.reshape(len(X_patches), -1)
X_patches_resampled, y_patches_resampled = smote.fit_resample(X_patches_reshaped, y_patches)
X_patches = X_patches_resampled.reshape(-1, config["num_channels"], config["patch_size"] * 2, config["patch_size"] * 2)
y_patches = y_patches_resampled

# Update original_indices to match the resampled dataset
num_original = len(original_indices)
num_resampled = len(y_patches)
new_original_indices = np.full(num_resampled, -1, dtype=np.int32)
new_original_indices[:num_original] = original_indices
original_indices = new_original_indices

logging.info(f"Post-SMOTE counts: {dict(zip(*np.unique(y_patches, return_counts=True)))}")
logging.info(f"Memory after SMOTE: {process.memory_info().rss / 1024**2:.2f} MB")
print("Dataset balanced!")

# Step 8: Compute Class Weights with Adjusted Emphasis
y_patches = y_patches.astype(int)
class_weights = compute_class_weight('balanced', classes=np.unique(y_patches), y=y_patches)
custom_weights = class_weights.copy()
custom_weights[1] = class_weights[1] * 3.0
custom_weights[2] = class_weights[2] * 2.5
custom_weights[4] = class_weights[4] * 0.9
class_weights_tensor = torch.tensor(custom_weights, dtype=torch.float32).to(device)
logging.info(f"Adjusted class weights: {custom_weights.tolist()}")

# Step 9: Prepare Data for Training
X_patches_tensor = torch.tensor(X_patches, dtype=torch.float32)
y_patches_tensor = torch.tensor(y_patches, dtype=torch.long)

# Step 10: Cross-Validation Setup (3-fold)
kf = KFold(n_splits=3, shuffle=True, random_state=42)
num_epochs = config["epochs"]
best_avg_val_loss = float('inf')
all_y_true = []
all_y_pred = []
fold_train_loss_history = []
fold_val_loss_history = []
fold_train_acc_history = []
fold_val_acc_history = []

start_time = datetime.now()
logging.info(f"Training started at: {start_time}")

for fold, (train_indices, val_indices) in enumerate(kf.split(range(len(X_patches)))):
    print(f"\nFold {fold + 1}/3")
    logging.info(f"Starting fold {fold + 1}")

    # Filter synthetic samples from validation
    real_val_mask = original_indices[val_indices] < len(filtered_dataset)
    val_indices = val_indices[real_val_mask]
    if len(val_indices) < 10:
        logging.warning(f"Fold {fold + 1}: Too few real validation samples ({len(val_indices)})")
        continue

    train_dataset = TensorDataset(X_patches_tensor[train_indices], y_patches_tensor[train_indices])
    val_dataset = TensorDataset(X_patches_tensor[val_indices], y_patches_tensor[val_indices])
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Step 11: Define the Model with Spatial Dropout
    model = models.resnet18(weights='IMAGENET1K_V1')
    for name, param in model.named_parameters():
        if "layer2" not in name and "layer3" not in name and "layer4" not in name and "fc" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    model.conv1 = nn.Conv2d(config["num_channels"], 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model_spatial_dropout = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        SpatialDropout(drop_prob=0.3)
    )
    model.conv1 = model_spatial_dropout
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(config["dropout_rate"]),
        nn.BatchNorm1d(num_ftrs),
        nn.Linear(num_ftrs, config["num_classes"])
    )
    model = model.to(device)

    # Step 12: Define Loss, Optimizer, Scheduler
    criterion = FocalLoss(gamma=2.5, alpha=class_weights_tensor, label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=config["initial_lr"], weight_decay=config["weight_decay"])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

    best_val_loss = float('inf')
    trials = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    # Step 13: Training Loop
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        val_losses_batch = []
        val_preds_counts = {i: 0 for i in range(config["num_classes"])}
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_losses_batch.append(loss.item())
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                for pred in preds.cpu().numpy():
                    val_preds_counts[pred] += 1
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_loss_std = np.std(val_losses_batch) if val_losses_batch else 0
        if val_loss_std > 0.05:
            logging.warning(f"Epoch {epoch + 1}: High validation loss variability (std={val_loss_std:.4f}), consider more regularization")

        scheduler.step(val_loss)

        print(f"Fold {fold + 1}, Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        logging.info(f"Fold {fold + 1}, Epoch {epoch + 1}: Train Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        logging.info(f"Fold {fold + 1}, Epoch {epoch + 1}: Validation prediction counts: {val_preds_counts}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trials = 0
            if val_loss < best_avg_val_loss:
                best_avg_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)
                logging.info(f"Saved best model at {model_save_path}")
        else:
            trials += 1
            if trials >= config["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

    # Collect predictions
    model.eval()
    fold_y_true, fold_y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            fold_y_true.extend(labels.cpu().numpy())
            fold_y_pred.extend(preds.cpu().numpy())
    
    print(f"Fold {fold + 1} Classification Report:")
    print(classification_report(fold_y_true, fold_y_pred, zero_division=0))
    logging.info(f"Fold {fold + 1} Classification Report:\n{classification_report(fold_y_true, fold_y_pred, zero_division=0)}")
    all_y_true.extend(fold_y_true)
    all_y_pred.extend(fold_y_pred)

    fold_train_loss_history.append(train_losses)
    fold_val_loss_history.append(val_losses)
    fold_train_acc_history.append(train_accs)
    fold_val_acc_history.append(val_accs)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold + 1} Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config["log_dir"], f"fold_{fold + 1}_loss_v24_final.png"))
    plt.close()

# Step 14: Plot Average Learning Curves
avg_train_loss = np.mean([history[:min(len(h) for h in fold_train_loss_history)] for history in fold_train_loss_history], axis=0)
avg_val_loss = np.mean([history[:min(len(h) for h in fold_val_loss_history)] for history in fold_val_loss_history], axis=0)
avg_train_acc = np.mean([history[:min(len(h) for h in fold_train_acc_history)] for history in fold_train_acc_history], axis=0)
avg_val_acc = np.mean([history[:min(len(h) for h in fold_val_acc_history)] for history in fold_train_acc_history], axis=0)

epochs_range = range(1, len(avg_train_loss) + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, avg_train_loss, label='Training Loss')
plt.plot(epochs_range, avg_val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Average Training and Validation Loss Across Folds')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(config["log_dir"], "train_val_loss_v24_final.png"))
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, avg_train_acc, label='Training Accuracy')
plt.plot(epochs_range, avg_val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Average Training and Validation Accuracy Across Folds')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(config["log_dir"], "train_val_accuracy_v24_final.png"))
plt.close()

# Step 15: Aggregated Evaluation
print("Aggregated Classification Report Across All Folds:")
print(classification_report(all_y_true, all_y_pred, zero_division=0))
print("Aggregated Confusion Matrix Across All Folds:")
print(confusion_matrix(all_y_true, all_y_pred))
logging.info(f"Aggregated Classification Report:\n{classification_report(all_y_true, all_y_pred, zero_division=0)}")
logging.info(f"Aggregated Confusion Matrix:\n{confusion_matrix(all_y_true, all_y_pred)}")

# Step 16: Save Validation Points
val_indices_last_fold = list(kf.split(range(len(X_patches))))[-1][1]
real_val_mask = original_indices[val_indices_last_fold] < len(filtered_dataset)
val_indices_last_fold = val_indices_last_fold[real_val_mask]
val_original_indices = original_indices[val_indices_last_fold]
valid_val_indices = np.where(val_original_indices != -1)[0]
val_original_indices = val_original_indices[valid_val_indices]
val_data = filtered_dataset.dataset.data.iloc[val_original_indices].copy()
val_data['Predicted_Class'] = np.array(all_y_pred)[-len(val_data):]
val_data['Is_Synthetic'] = (val_original_indices >= len(filtered_dataset)).astype(int)
val_data.to_csv(os.path.join(config["log_dir"], "validation_points_v24_final.csv"), index=False)
print("Validation points saved to 'validation_points_v24_final.csv'")

# Step 17: Geospatial Error Analysis
val_df = pd.read_csv(os.path.join(config["log_dir"], "validation_points_v24_final.csv"))
val_gdf = gpd.GeoDataFrame(
    val_df, geometry=gpd.points_from_xy(val_df['longitude'], val_df['latitude']),
    crs="EPSG:4326"
)
val_gdf['Misclassified'] = val_gdf['Class'] != val_gdf['Predicted_Class']
plt.figure(figsize=(10, 8))
val_gdf[val_gdf['Misclassified']].plot(
    column='Class', categorical=True, legend=True,
    markersize=20, cmap='tab10', alpha=0.6
)
plt.title("Misclassified Points by True Class")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.savefig(os.path.join(config["log_dir"], "misclassification_map_v24_final.png"))
plt.close()
logging.info("Geospatial error analysis completed and saved to 'misclassification_map_v24_final.png'")

end_time = datetime.now()
logging.info(f"Training completed at: {end_time}")
logging.info(f"Total runtime: {end_time - start_time}")
print(f"Model saved to {model_save_path}")
logging.info(f"Training completed. Model saved to {model_save_path}")