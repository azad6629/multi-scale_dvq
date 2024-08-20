import os
from pathlib import Path

DATA_BASE_DIR = Path("/workspace/DATASETS/XRAY_datasets/")
ISIC_BASE_DIR = Path("/workspace/DATASETS/SKIN/")



# #############################################
# NIH constants
# #############################################
NIH_CXR_DATA_DIR       = DATA_BASE_DIR / "NIH_Chest-Xray-14"
NIH_DATA_ENTRY_CSV     = NIH_CXR_DATA_DIR / "Data_Entry_2017.csv"
NIH_ORIGINAL_TRAIN_TXT = NIH_CXR_DATA_DIR / "train_val_list.txt"
NIH_ORIGINAL_TEST_TXT  = NIH_CXR_DATA_DIR / "test_list.txt"
NIH_TRAIN_CSV          = NIH_CXR_DATA_DIR / "train.csv"
NIH_TEST_CSV           = NIH_CXR_DATA_DIR / "test.csv"
NIH_TRAIN_BBOX_CSV     = NIH_CXR_DATA_DIR / "train_box.csv"
NIH_TEST_BBOX_CSV      = NIH_CXR_DATA_DIR / "test_box.csv"


NIH_PERT_TRAIN_CSV10 = NIH_CXR_DATA_DIR / "10_perturb_train_data.csv"
NIH_PERT_TRAIN_CSV20 = NIH_CXR_DATA_DIR / "20_perturb_train_data.csv"
NIH_PERT_TRAIN_CSV30 = NIH_CXR_DATA_DIR / "30_perturb_train_data.csv"
NIH_PERT_TRAIN_CSV40 = NIH_CXR_DATA_DIR / "40_perturb_train_data.csv"
NIH_PERT_TRAIN_CSV50 = NIH_CXR_DATA_DIR / "50_perturb_train_data.csv" 
    
NIH_PERT_TEST_CSV    = NIH_CXR_DATA_DIR / "perturb_test_data.csv"


NIH_PATH_COL = "Path"
NIH_TASKS = [
             "Atelectasis" ,
             "Cardiomegaly", 
             "Consolidation", 
             "Infiltration", 
             "Pneumothorax", 
             "Edema", 
             "Emphysema", 
             "Fibrosis", 
             "Effusion", 
             "Pneumonia", 
             "Pleural_Thickening", 
             "Nodule", 
             "Mass", 
             "Hernia",
             "No Finding"
            ]

NIH_CXR8_TASKS = [
                "Atelectasis",
                "Cardiomegaly",
                "Effusion",
                "Infiltration",
                "Mass",
                "Nodule",
                "Pneumonia",
                "Pneumothorax"
            ]




# #############################################
# ISIC_2018 constants
# #############################################
ISIC_DATA_DIR   = ISIC_BASE_DIR / "ISIC_18"
ISIC_TRAIN_DIR  = ISIC_DATA_DIR / "ISIC2018_Task3_Training_Input"
ISIC_VALID_DIR  = ISIC_DATA_DIR / "ISIC2018_Task3_Validation_Input"
ISIC_TEST_DIR   = ISIC_DATA_DIR / "ISIC2018_Task3_Test_Input"
ISIC_TRAIN_CSV  = ISIC_DATA_DIR / "train.csv"
ISIC_VALID_CSV  = ISIC_DATA_DIR / "valid.csv"
ISIC_TEST_CSV   = ISIC_DATA_DIR / "test.csv"

