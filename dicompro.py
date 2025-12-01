import os
import shutil
import hashlib
import pydicom
from pathlib import Path
from collections import defaultdict

def is_valid_dicom(file_path):
    try:
        pydicom.dcmread(file_path, stop_before_pixels=True)
        return True
    except:
        return False


def get_safe_filename(name):
    return "".join(c if c.isalnum() else "_" for c in str(name))


def organize_dicom_files(src_dir, dest_dir="organized_dicom"):
    Path(dest_dir).mkdir(exist_ok=True)

    patient_index = defaultdict(lambda: defaultdict(list))
    error_files = []
    processed_count = 0

    for root, _, files in os.walk(src_dir):
        for filename in files:
            file_path = os.path.join(root, filename)

            if not is_valid_dicom(file_path):
                error_files.append(file_path)
                continue

            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)

                patient_id = getattr(ds, 'AccessionNumber', 'Unknown_Patient')
                patient_name = getattr(ds, 'PatientName', 'Unnamed')

                if patient_id == 'Unknown_Patient':
                    patient_id = f"Patient_{hashlib.md5(str(patient_name).encode()).hexdigest()[:8]}"

                series_id = getattr(ds, 'SeriesInstanceUID', None)
                if not series_id:
                    series_id = f"{getattr(ds, 'SeriesNumber', 'SN')}-" \
                                f"{getattr(ds, 'SeriesDate', 'SD')}-" \
                                f"{getattr(ds, 'SeriesDescription', '')}"
                    series_id = hashlib.md5(series_id.encode()).hexdigest()[:12]

                patient_index[patient_id][series_id].append(file_path)
                processed_count += 1

            except Exception as e:
                error_files.append(file_path)
                continue

    for patient_id, series_data in patient_index.items():
        patient_dir = os.path.join(dest_dir, f"{get_safe_filename(patient_id)}")
        Path(patient_dir).mkdir(exist_ok=True)

        with open(os.path.join(patient_dir, "_patient_info.txt"), "w") as f:
            f.write(f"Patient ID: {patient_id}\nTotal Series: {len(series_data)}")

        for series_id, files in series_data.items():
            sample_ds = pydicom.dcmread(files[0], stop_before_pixels=True)
            series_desc = get_safe_filename(getattr(sample_ds, 'SeriesDescription', 'Unnamed_Series'))
            modality = getattr(sample_ds, 'Modality', 'OT')

            series_dir = os.path.join(
                patient_dir,
                f"{modality}_{series_desc}"
            )
            Path(series_dir).mkdir(exist_ok=True)

            for idx, src_path in enumerate(files, 1):
                dst_filename = f"{modality}_{idx:04d}.dcm"
                shutil.copy2(src_path, os.path.join(series_dir, dst_filename))

if __name__ == "__main__":
    organize_dicom_files(
        src_dir="./Dicom_folder",
        dest_dir="./new_dataset"
    )
