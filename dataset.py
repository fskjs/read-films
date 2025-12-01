import os
import glob
import numpy as np
import torch
import pydicom
import scipy.ndimage
import pandas as pd
from skimage.transform import resize

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CT3DDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, id_label_csv, max_slices=512,
                 voxel_size=(2.0, 0.8, 0.8), cache_dir=None,
                 img_size=(128, 128), verbose=False,
                 aux_feature_size=50):
        self.root_dir = root_dir
        self.id_label_csv = id_label_csv
        self.max_slices = max_slices
        self.voxel_size = np.array(voxel_size)
        self.cache_dir = cache_dir
        self.img_size = img_size
        self.verbose = verbose
        self.logger = logging.getLogger('CT3DDataset')

        self.height, self.width = self.img_size

        self.aux_feature_size = aux_feature_size
        self._load_id_labels_and_features()

        self._scan_patient_dirs()

    def _load_id_labels_and_features(self):
        self.all_id = []
        self.id_labels = {}
        self.id_features = {}
        try:
            df = pd.read_csv(self.id_label_csv)

            if 'id' in df.columns:
                for i, row in df.iterrows():
                    patient_id = str(row['id'])
                    label = int(row.get('label', -1))
                    features = row.drop(['id', 'label']).values.astype(np.float32)

                    if len(features) != self.aux_feature_size:
                        print(f"Warning:  {patient_id} Feature dimension error: {len(features)} != Expected{self.aux_feature_size}")
                        continue

                    self.id_labels[patient_id] = label
                    self.id_features[patient_id] = features
                    self.all_id.append(patient_id)
            else:
                for i in range(len(df)):
                    row = df.iloc[i]
                    patient_id = str(row[0])
                    label = int(row[1])

                    features = row.values[2:27].astype(np.float32)

                    if len(features) != self.aux_feature_size:
                        print(f"Warning {patient_id} Feature dimension error: {len(features)}")
                        continue

                    self.id_labels[patient_id] = label
                    self.id_features[patient_id] = features
                    self.all_id.append(patient_id)

        except Exception as e:
            self.logger.error(f"Error loading ID tags and feature files: {str(e)}")
            raise

    def _scan_patient_dirs(self):
        self.patient_data = []

        patient_dirs = list(glob.iglob(os.path.join(self.root_dir, '*')))

        valid_count = 0
        invalid_ids = []

        for p_dir in patient_dirs:
            if not os.path.isdir(p_dir):
                continue

            patient_id = os.path.basename(p_dir).strip()

            if patient_id not in self.id_labels:
                if patient_id != '':
                    invalid_ids.append(patient_id)
                continue

            label = self.id_labels[patient_id]
            features = self.id_features[patient_id]
            if label == -1:
                self.logger.warning(f" {patient_id} has an invalid label (-1), skipped")
                continue

            dicom_paths = []
            for root, _, files in os.walk(p_dir):
                for file in files:
                    if file.lower().endswith('.dcm'):
                        dicom_paths.append(os.path.join(root, file))

            if not dicom_paths:
                self.logger.warning(f" {patient_id} has no DICOM file，skipped")
                continue

            cache_file = None
            if self.cache_dir:
                cache_file = os.path.join(self.cache_dir, f"{patient_id}.npy")
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)

            self.patient_data.append({
                'id': patient_id,
                'dir': p_dir,
                'dicom_paths': sorted(dicom_paths),
                'label': label,
                'cache_file': cache_file,
                'features': features
            })
            valid_count += 1

    def __len__(self):
        return len(self.patient_data)

    def check_dimensions(self):
        if not self.patient_data:
            print("No patient data available for examination")
            return False

        all_shapes = []

        for i in range(min(len(self), 10)):
            vol = self._load_patient_volume(i)
            vol_shape = vol.shape

            if isinstance(vol_shape, tuple):
                all_shapes.append(vol_shape)
            else:
                print(f"Error: In indexing {i} found invalid shapes {vol_shape}")

            if i > 0 and all_shapes[i] != all_shapes[0]:
                return False

        if len(set(all_shapes)) == 1:
            print(f"The sample size was consistent across all examinations: {all_shapes[0]}")
            return True
        return False

    def _load_and_stack_dicoms(self, dicom_paths):
        slices = []
        positions = []
        slices_info = []
        position = 0.0

        for path in dicom_paths:
            try:
                ds = pydicom.dcmread(path)
                img = ds.pixel_array.astype(np.float32)

                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    img = img * ds.RescaleSlope + ds.RescaleIntercept

                slices.append(img)

                position = position + 1
                positions.append(position)

                slices_info.append({
                    'SliceThickness': float(getattr(ds, 'SliceThickness', 1.0)),
                    'PixelSpacing': list(map(float, getattr(ds, 'PixelSpacing', [1.0, 1.0])))
                })

            except Exception as e:
                print(f"Error loading DICOM file {path}: {str(e)}")
                continue

        if len(slices) < self.max_slices // 10:
            raise ValueError(f"Insufficient DICOM slices: {len(slices)} < {self.max_slices // 10}")

        if not slices_info:
            raise ValueError("No valid metadata is available")

        z_thickness = np.mean([info.get('SliceThickness', 1.0) for info in slices_info])
        pixel_spacing = np.mean([info.get('PixelSpacing', [1.0, 1.0]) for info in slices_info], axis=0)

        sorted_indices = np.argsort(positions)
        sorted_slices = [slices[i] for i in sorted_indices]

        volume = np.stack(sorted_slices, axis=0)

        return volume, {
            'z_thickness': z_thickness,
            'pixel_spacing': pixel_spacing,
            'num_slices': len(slices)
        }

    def _resample_volume(self, volume, metadata):
        if volume.size == 0:
            raise ValueError("Empty volumes cannot be resampled")

        current_z_thickness = metadata.get('z_thickness', 5.0)
        current_pixel_spacing = metadata.get('pixel_spacing', [1.0, 1.0])

        if current_z_thickness <= 0:
            current_z_thickness = 5.0

        z_factor = self.voxel_size[0] / current_z_thickness
        y_factor = self.voxel_size[1] / current_pixel_spacing[0]
        x_factor = self.voxel_size[2] / current_pixel_spacing[1]

        new_depth = max(1, int(volume.shape[0] * z_factor))
        new_height = max(1, int(volume.shape[1] * y_factor))
        new_width = max(1, int(volume.shape[2] * x_factor))

        try:
            zoom_factors = [z_factor, y_factor, x_factor]
            resampled_vol = scipy.ndimage.zoom(
                volume,
                zoom_factors,
                order=1,
                mode='nearest'
            )
            return resampled_vol
        except Exception as e:
            return self._resample_manual(volume, z_factor, new_depth, new_height, new_width)

    def _resample_manual(self, volume, z_factor, new_depth, new_height, new_width):
        resampled_vol = np.zeros((new_depth, new_height, new_width), dtype=volume.dtype)

        for z in range(new_depth):
            src_z = min(int(z / z_factor), volume.shape[0] - 1)
            src_slice = volume[src_z]

            if len(src_slice.shape) == 2:
                resampled_vol[z] = resize(
                    src_slice,
                    (new_height, new_width),
                    order=1,
                    preserve_range=True
                )
            else:
                resampled_vol[z, :, :] = 0

        return resampled_vol

    def _normalize_volume1(self, volume):
        p1 = np.percentile(volume, 1)
        p99 = np.percentile(volume, 99)
        if p99 > p1:
            volume = np.clip(volume, p1, p99)

        vmin = np.min(volume)
        vmax = np.max(volume)
        if vmax > vmin:
            volume = (volume - vmin) / (vmax - vmin)
        return volume


    def _normalize_volume(self, volume):
        hu_min = -150.0
        hu_max = 1000.0
        volume = np.clip(volume, hu_min, hu_max)
        volume = (volume - hu_min) / (hu_max - hu_min)
        return volume

    def _load_patient_volume(self, patient_idx):
        patient = self.patient_data[patient_idx]

        if self.cache_dir and os.path.exists(patient['cache_file']):
            if self.verbose:
                print(f"loading cache: {patient['cache_file']}")
            volume = np.load(patient['cache_file'])
            return volume

        try:
            volume, metadata = self._load_and_stack_dicoms(patient['dicom_paths'])

            volume = self._resample_volume(volume, metadata)

            volume = self._normalize_volume(volume)
        except Exception as e:
            volume = np.zeros((self.max_slices, self.height, self.width))

        if volume.shape[0] > self.max_slices:
            indices = np.linspace(0, volume.shape[0] - 1, self.max_slices, dtype=int)
            volume = volume[indices]
        elif volume.shape[0] < self.max_slices:
            pad_after = self.max_slices - volume.shape[0] - 0
            volume = np.pad(
                volume,
                ((0, pad_after), (0, 0), (0, 0)),
                'constant',
                constant_values=0
            )

        if volume.shape[1] != self.height or volume.shape[2] != self.width:
            resampled = np.zeros((volume.shape[0], self.height, self.width))
            for i in range(volume.shape[0]):
                resampled[i] = resize(
                    volume[i],
                    (self.height, self.width),
                    anti_aliasing=True,
                    preserve_range=True
                )
            volume = resampled

        if len(volume.shape) != 3:
            if len(volume.shape) == 2:
                volume = volume[np.newaxis, ...]
            elif len(volume.shape) > 3:
                volume = volume.squeeze()
                if len(volume.shape) > 3:
                    volume = volume[..., 0]
            if len(volume.shape) != 3:
                volume = np.zeros((self.max_slices, self.height, self.width))

        if self.cache_dir:
            if self.verbose:
                np.save(patient['cache_file'], volume)

        return volume

    def _safe_load_dicom(self, path):
        try:
            if not os.path.exists(path):
                self.logger.warning(f"DICOM File does not exist: {path}")
                return None

            ds = pydicom.dcmread(path)

            if 'PixelData' not in ds:
                self.logger.warning(f"DICOM Missing pixel data: {path}")
                return None

            img = ds.pixel_array.copy().astype(np.float32)

            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

            if np.isnan(img).any() or np.isinf(img).any():
                self.logger.warning(f"detected NaN/Inf: {path}")

            if any([isinstance(pixel, str) for pixel in img.flatten()]):
                self.logger.error(f"The pixel value contains a string: {path}")
                return None

            return img
        except Exception as e:
            self.logger.error(f"Failed to load DICOM {path}: {str(e)}")
            return None

    @staticmethod
    def collate_fn(batch):
        volumes = torch.stack([item['volume'] for item in batch], dim=0)  # [batch, 1, depth, H, W]
        labels = torch.tensor([item['label'] for item in batch])
        depths = torch.cat([item['depth'] for item in batch], dim=0)
        features = torch.stack([item['features'] for item in batch], dim=0)  # [batch, 24]
        ids = [item['id'] for item in batch]

        return {
            'volume': volumes,
            'label': labels,
            'depth': depths,
            'features': features,
            'id': ids
        }

    def __getitem__(self, idx):
        try:
            volume = self._load_patient_volume(idx)
            volume = np.expand_dims(volume, axis=0)
            volume_tensor = torch.tensor(volume, dtype=torch.float32)
            volume_depth = volume_tensor.shape[1]

            aux_features = self.patient_data[idx]['features']
            aux_tensor = torch.tensor(aux_features, dtype=torch.float32)

            if self.verbose and idx % 10 == 0:
                print(f"loading  {self.patient_data[idx]['id']}: "
                      f"shape={volume_tensor.shape}, label={self.patient_data[idx]['label']}")

            sample_id = self.patient_data[idx]['id']

            return {
                'volume': volume_tensor,
                'features': aux_tensor,
                'label': self.patient_data[idx]['label'],
                'depth': torch.tensor([volume_depth], dtype=torch.long),
                'id': sample_id
            }

        except Exception as e:
            print(f"Process indexes {idx} error: {str(e)}")
            return {
                'volume': torch.zeros(1, self.max_slices, self.height, self.width),
                'label': -1,
                'depth': torch.tensor([self.max_slices], dtype=torch.long),
                'features': torch.zeros(self.aux_feature_size),
                'id': 'error_id'
            }


