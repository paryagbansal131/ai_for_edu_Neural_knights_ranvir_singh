# import os
# import csv
# import shutil

# # ===================== PATHS =====================
# LABEL_FILE = "dataset/train_labels.xlsx"          # your CSV file
# SOURCE_DIR = "frames_100x100"      # folder with *_2fps folders
# DEST_0 = "class_0"
# DEST_1 = "class_1"

# # Create output folders
# os.makedirs(DEST_0, exist_ok=True)
# os.makedirs(DEST_1, exist_ok=True)

# # ===================== READ LABELS =====================
# labels = {}

# with open(LABEL_FILE, newline='', encoding='latin-1') as f:
#     reader = csv.DictReader(f)

#     for row in reader:
#         filename = row["filename"].strip()
#         label = float(row["label"])

#         # Binary classification rule
#         binary_label = 0 if label <= 0.33 else 1
#         labels[filename] = binary_label

# print(f"Loaded {len(labels)} labels")

# # ===================== PROCESS FOLDERS =====================
# for folder in os.listdir(SOURCE_DIR):
#     folder_path = os.path.join(SOURCE_DIR, folder)

#     if not os.path.isdir(folder_path):
#         continue

#     # Convert:
#     # subject_20_Vid_6_2fps → subject_20_Vid_6
#     base_name = folder.replace("_2fps", "")

#     # Try all possible extensions
#     matched_file = None
#     for ext in [".avi", ".MP4", ".mp4", ".webm", ".wmv"]:
#         candidate = base_name + ext
#         if candidate in labels:
#             matched_file = candidate
#             break

#     if matched_file is None:
#         print(f"⚠️ No label found for {folder}")
#         continue

#     target_dir = DEST_0 if labels[matched_file] == 0 else DEST_1
#     dest_path = os.path.join(target_dir, folder)

#     # Copy folder
#     shutil.copytree(folder_path, dest_path, dirs_exist_ok=True)

#     print(f"✔ {folder} → {'class_0' if labels[matched_file] == 0 else 'class_1'}")

# print("✅ Classification completed successfully")
# import os
# import shutil
# import pandas as pd

# BASE_DIR = "dataset"
# FRAMES_DIR = os.path.join("frames_100x100")
# LABEL_FILE = os.path.join(BASE_DIR, "train_labels.xlsx")
# OUTPUT_DIR = os.path.join(BASE_DIR, "output_images_complete")

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Load Excel labels
# df = pd.read_excel(LABEL_FILE)

# VIDEO_COL = df.columns[0]   # video name
# LABEL_COL = df.columns[1]   # label

# # Map video → label
# video_to_label = dict(zip(df[VIDEO_COL].astype(str), df[LABEL_COL].astype(str)))

# print("📊 Total labels loaded:", len(video_to_label))

# # ONLY process folders inside frames_100x100
# for folder in os.listdir(FRAMES_DIR):
#     src = os.path.join(FRAMES_DIR, folder[:-5])
#     if not os.path.isdir(src):
#         continue  # ignore files

#     if folder not in video_to_label:
#         print(f"⚠️ Label missing for: {folder}")
#         continue

#     label = video_to_label[folder]

#     label_dir = os.path.join(OUTPUT_DIR, label)
#     os.makedirs(label_dir, exist_ok=True)

#     dest = os.path.join(label_dir, folder)

#     shutil.move(src, dest)

#     print(f"✅ {folder} → label {label}")

# print("🎉 DONE: ONLY frames_100x100 processed safely")
import os
import shutil
import pandas as pd

# ===================== PATHS =====================
EXCEL_FILE = "dataset/train_labels.xlsx"        # your Excel file
SOURCE_DIR = "frames_100x100"
DEST_0 = "label0"
DEST_1 = "label1"

os.makedirs(DEST_0, exist_ok=True)
os.makedirs(DEST_1, exist_ok=True)

# ===================== READ EXCEL (SKIP HEADER) =====================
df = pd.read_excel(EXCEL_FILE)   # header is automatically handled

labels = {}

for _, row in df.iterrows():
    filename = str(row.iloc[0]).strip()
    label = float(row.iloc[1])

    binary_label = 0 if label <= 0.33 else 1
    labels[filename] = binary_label

print(f"✅ Loaded {len(labels)} labels from Excel")

# ===================== PROCESS IMAGE FOLDERS =====================
for folder in os.listdir(SOURCE_DIR):
    folder_path = os.path.join(SOURCE_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    # subject_1_Vid_5_2fps → subject_1_Vid_5
    base_name = folder.replace("_2fps", "")

    matched_file = None
    for ext in [".avi", ".AVI", ".mp4", ".MP4", ".webm", ".wmv"]:
        candidate = base_name + ext
        if candidate in labels:
            matched_file = candidate
            break

    if matched_file is None:
        print(f"⚠️ No label found for {folder}")
        continue

    target_dir = DEST_0 if labels[matched_file] == 0 else DEST_1
    dest_path = os.path.join(target_dir, folder)

    shutil.copytree(folder_path, dest_path, dirs_exist_ok=True)

    print(f"✔ {folder} → {'label0' if labels[matched_file] == 0 else 'label1'}")

print("🎯 DONE: All folders classified successfully")