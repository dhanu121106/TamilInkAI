# TamilInk AI — Mini Project Demo

**Project title:** TamilInk AI: Automated Restoration and Recognition of Handwritten Tamil Documents

This package provides a working restoration demo and an optional Tamil OCR integration. It is a prototype, not a fully trained handwritten-Tamil recognition model.

## Run on Windows

1. Install Python 3.10 or 3.11.
2. Extract the ZIP and open this folder in VS Code.
3. In the terminal:
   ```powershell
   py -3.11 -m venv venv
   .\venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Tamil OCR setup (optional)
The app uses Tesseract OCR with the Tamil language pack (`tam`) if installed. Install Tesseract OCR for Windows and ensure the Tamil traineddata file is installed. If Tesseract is not on PATH, set the executable path in PowerShell before running Streamlit, for example:
```powershell
$env:TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
streamlit run app.py
```
The exact installation path may differ. In Tesseract, the Tamil language code is `tam`. General OCR is not guaranteed to accurately recognize handwritten Tamil.

## Dataset sample
The included `samples/synthetic_tamil_demo.png` is a **synthetic sample created only to demonstrate the restoration UI**; it is not represented as a real dataset image. For genuine handwritten Tamil word images, use the official IIIT-INDIC-HW-WORDS Tamil dataset page:
https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-hw-words
Download and use its sample according to the dataset's access and usage terms. The official dataset contains handwritten Tamil word images with corresponding transcriptions.

## Workflow
Upload image → grayscale → denoising → CLAHE contrast enhancement → adaptive/Otsu thresholding → download restored image → optional Tamil OCR.

## Important limitations
- Restoration uses conventional image processing, not a learned restoration network.
- Tesseract OCR is a general OCR engine, not a custom-trained Tamil handwriting model.
- Old, faded, connected, cursive, or degraded Tamil handwriting may be misrecognized.
- Never report a fabricated OCR result or claim accuracy without evaluation on labeled data.
