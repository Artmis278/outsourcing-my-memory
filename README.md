# Outsourcing My Memory

A local AI photo search app built with CLIP and Streamlit.

Enter a photo folder and click **Index folder** to index your images. No photo library is accessed automatically at startup.

Search using natural-language descriptions such as:

> snowy mountain at sunset

CLIP finds images based on visual similarity. It is not an exact object detector or OCR tool.

## Privacy

Original photos are only opened for reading.

All processing happens locally. The app does not upload your photos to the cloud.

The app files, SQLite index, and downloaded CLIP weights are stored locally under the project directory.

## Indexing

Indexing is incremental and can resume after interruption.

Supported formats:

- JPEG
- PNG
- WebP
- BMP
- TIFF

RAW and HEIC are not currently supported.

The first indexing operation downloads the CLIP model weights.

## Running the app

From PowerShell:

```powershell
./launch.ps1
```
## Requirements

- Python 3.10+
- Windows
- Dependencies listed in `requirements-lock.txt`
