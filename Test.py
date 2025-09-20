import sys
import os
import numpy as np
import torch
import torch.nn as nn
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import Polygon
import pandas as pd
from unittest.mock import patch
import logging
from sklearn.preprocessing import StandardScaler
import pytest

# Setup logging
test_log_dir = "test_logs"
os.makedirs(test_log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, filename=os.path.join(test_log_dir, "test_log.log"), format='%(asctime)s - %(message)s')

# Add path to fold3_2024.py dynamically
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from the renamed module
from fold3_2024 import ChannelAttention, SpatialAttention, CBAM, SpatialDropout, LandUseDataset, ResampledDataset, config, model

# Create synthetic test data
def create_synthetic_data():
    # Synthetic Landsat-8 image (5 bands, 100x100 pixels)
    transform = from_origin(33.0, -13.0, 30, 30)  # 30m resolution
    landsat_data = np.random.rand(5, 100, 100).astype(np.float32) * 1000  # Random values
    landsat_data[0, 50:60, 50:60] = 0  # Simulate no-data
    landsat_profile = {
        'driver': 'GTiff', 'height': 100, 'width': 100, 'count': 5, 'dtype': 'float32',
        'crs': 'EPSG:4326', 'transform': transform, 'nodata': 0
    }
    os.makedirs("test_data", exist_ok=True)
    with rasterio.open('test_data/synthetic_landsat.tif', 'w', **landsat_profile) as dst:
        dst.write(landsat_data)
        dst.descriptions = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1']

    # Synthetic DEM raster
    dem_data = np.random.rand(100, 100).astype(np.float32) * 100  # Random elevation
    dem_profile = {
        'driver': 'GTiff', 'height': 100, 'width': 100, 'count': 1, 'dtype': 'float32',
        'crs': 'EPSG:4326', 'transform': transform, 'nodata': 0
    }
    with rasterio.open('test_data/synthetic_dem.tif', 'w', **dem_profile) as dst:
        dst.write(dem_data, 1)

    # Synthetic roads raster
    roads_data = np.random.randint(0, 2, (100, 100)).astype(np.float32)  # Binary road mask
    roads_profile = {
        'driver': 'GTiff', 'height': 100, 'width': 100, 'count': 1, 'dtype': 'float32',
        'crs': 'EPSG:4326', 'transform': transform, 'nodata': 0
    }
    with rasterio.open('test_data/synthetic_roads.tif', 'w', **roads_profile) as dst:
        dst.write(roads_data, 1)

    # Synthetic ground truth points
    points = pd.DataFrame({
        'longitude': [33.0, 33.01, 33.02, 33.03, 33.04],
        'latitude': [-13.0, -13.01, -13.02, -13.03, -13.04],
        'Class': [0, 1, 2, 3, 4]
    })
    points.to_csv('test_data/synthetic_ground_truth.csv', index=False)

    # Synthetic SMOTE-resampled data
    smote_features = np.random.rand(10, config["num_channels"] * config["patch_size"] * 2 * config["patch_size"]).astype(np.float32)
    smote_labels = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 4])
    return smote_features, smote_labels

# Mock model for testing
class MockModel(nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.fc = nn.Linear(512, config["num_classes"])  # Mock ResNet output

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten
        return self.fc(x)

def test_channel_attention():
    # Test ChannelAttention module
    ca = ChannelAttention(in_planes=64, ratio=16).to(torch.device("cpu"))
    x = torch.randn(1, 64, 32, 32)
    out = ca(x)
    assert out.shape == (1, 64, 1, 1), f"Expected shape (1, 64, 1, 1), got {out.shape}"
    assert torch.all(out >= 0) and torch.all(out <= 1), "Output not in sigmoid range"
    logging.info("ChannelAttention test passed")

def test_spatial_attention():
    # Test SpatialAttention module
    sa = SpatialAttention(kernel_size=7).to(torch.device("cpu"))
    x = torch.randn(1, 64, 32, 32)
    out = sa(x)
    assert out.shape == (1, 1, 32, 32), f"Expected shape (1, 1, 32, 32), got {out.shape}"
    assert torch.all(out >= 0) and torch.all(out <= 1), "Output not in sigmoid range"
    logging.info("SpatialAttention test passed")

def test_cbam():
    # Test CBAM module
    cbam = CBAM(in_planes=64, ratio=16, kernel_size=7).to(torch.device("cpu"))
    x = torch.randn(1, 64, 32, 32)
    out = cbam(x)
    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
    logging.info("CBAM test passed")

def test_spatial_dropout():
    # Test SpatialDropout module
    sd = SpatialDropout(drop_prob=0.2).to(torch.device("cpu"))
    sd.eval()  # Test eval mode
    x = torch.randn(1, 64, 32, 32)
    out = sd(x)
    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
    assert torch.equal(out, x), "Output changed in eval mode"
    sd.train()  # Test training mode
    out = sd(x)
    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
    logging.info("SpatialDropout test passed")

def test_land_use_dataset():
    # Create synthetic data
    smote_features, smote_labels = create_synthetic_data()
    
    # Test LandUseDataset
    dataset = LandUseDataset(
        image_path='test_data/synthetic_landsat.tif',
        dem_path='test_data/synthetic_dem.tif',
        roads_path='test_data/synthetic_roads.tif',
        ground_truth_csv='test_data/synthetic_ground_truth.csv',
        window_size=config["patch_size"]
    )
    
    # Fit scalers
    dataset.fit_scalers()
    
    # Test dataset length
    assert len(dataset) == 5, f"Expected 5 points, got {len(dataset)}"
    
    # Test patch retrieval
    patch, label = dataset[0]
    assert patch.shape == (config["num_channels"], config["patch_size"] * 2, config["patch_size"] * 2), \
        f"Expected shape ({config['num_channels']}, {config['patch_size']*2}, {config['patch_size']*2}), got {patch.shape}"
    assert isinstance(label, int) and 0 <= label < config["num_classes"], f"Invalid label: {label}"
    assert not torch.any(torch.isnan(patch)), "Patch contains NaN"
    logging.info("LandUseDataset test passed")

def test_resampled_dataset():
    # Create synthetic SMOTE data
    smote_features, smote_labels = create_synthetic_data()
    
    # Test ResampledDataset
    dataset = ResampledDataset(smote_features, smote_labels, config["num_channels"], config["patch_size"])
    assert len(dataset) == 10, f"Expected 10 samples, got {len(dataset)}"
    
    # Test data retrieval
    feature, label = dataset[0]
    assert feature.shape == (config["num_channels"], config["patch_size"] * 2, config["patch_size"] * 2), \
        f"Expected shape ({config['num_channels']}, {config['patch_size']*2}, {config['patch_size']*2}), got {feature.shape}"
    assert isinstance(label, np.integer) and 0 <= label < config["num_classes"], f"Invalid label: {label}"
    logging.info("ResampledDataset test passed")

def test_training_loop():
    # Create synthetic data
    smote_features, smote_labels = create_synthetic_data()
    
    # Mock dataset and dataloader
    dataset = ResampledDataset(smote_features, smote_labels, config["num_channels"], config["patch_size"])
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Mock model
    mock_model = MockModel().to(torch.device("cpu"))
    
    # Mock criterion and optimizer
    class_counts = np.bincount(smote_labels, minlength=config["num_classes"])
    class_weights = torch.tensor(1.0 / (class_counts + 1e-8), dtype=torch.float32).to(torch.device("cpu"))
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Test training step
    with patch('fold3_2024.model', mock_model):
        mock_model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(torch.device("cpu")), labels.to(torch.device("cpu"))
            outputs = mock_model(inputs)
            loss = criterion(outputs, labels)
            assert not torch.isnan(loss), "Loss is NaN"
            assert outputs.shape == (inputs.shape[0], config["num_classes"]), \
                f"Expected output shape ({inputs.shape[0]}, {config['num_classes']}), got {outputs.shape}"
            break  # Test one batch
    logging.info("Training loop test passed")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    print("All tests completed. Check test_logs/test_log.log for details.")