# BioVision-CellSegmenter — Task Tracker

## Phase 1: Environment Setup & Data Preparation
- [x] Acknowledge project & set up folder structure
- [x] Set up Python virtual environment (.gitignore, requirements, dirs)
- [x] Install dependencies (torch, opencv, smp, fastapi, tifffile)
- [x] Locate DSB 2018 dataset (447 train, 50 test)
- [x] Verify dataset structure — 256×256 grayscale TIFFs, instance masks
- [x] Create PyTorch `NucleiDataset` & `make_loaders` (256×256, binarized masks, train/val split)
- [x] Smoke-test: image shape [4,3,256,256], mask [4,1,256,256], unique {0,1}
- [x] Git init + first commit

## Phase 2: Deep Learning Model Training (PyTorch)
- [x] Initialize U-Net (`resnet34`) via SMP with ImageNet pretrained weights
- [x] Dice + BCE loss, Adam optimizer
- [x] 15 epoch training loop on full 447-image DSB 2018 training set
- [x] Model converged: val_loss=0.0842, val_dice=0.9201, val_iou=0.8562
- [x] Save `unet_nuclei.pth` + `training_history.json`

## Phase 3: CV Pipeline & Feature Extraction
- [ ] CLAHE pre-processing
- [ ] Inference → binary mask
- [ ] `cv2.findContours` outlines
- [ ] `regionprops` for count & avg area

## Phase 4: FastAPI Deployment
- [ ] `main.py` FastAPI app
- [ ] `POST /analyze` endpoint
- [ ] JSON response + base64 annotated image
