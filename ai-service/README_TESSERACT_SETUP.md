# Tesseract OCR Setup for Windows

## Why Tesseract?
Tesseract provides FREE, fast, local OCR without API costs. It's the primary OCR engine, with Gemini as fallback for low-confidence results.

## Installation Steps

### 1. Download Tesseract for Windows
Visit: https://github.com/UB-Mannheim/tesseract/wiki

Download the latest installer (e.g., `tesseract-ocr-w64-setup-5.3.3.20231005.exe`)

### 2. Install Tesseract
- Run the installer
- **IMPORTANT**: During installation, select "Add to PATH" option
- Default install location: `C:\Program Files\Tesseract-OCR`

### 3. Verify Installation
Open Command Prompt and run:
```bash
tesseract --version
```

You should see output like:
```
tesseract 5.3.3
 leptonica-1.83.1
  libgif 5.2.1 : libjpeg 8d (libjpeg-turbo 2.1.5.1) : libpng 1.6.40 : libtiff 4.5.1 : zlib 1.2.13 : libwebp 1.3.2 : libopenjp2 2.5.0
```

### 4. Set Environment Variable (If Not Auto-Added)
If `tesseract --version` doesn't work:

1. Press `Win + R`, type `sysdm.cpl`, press Enter
2. Go to "Advanced" tab → "Environment Variables"
3. Under "System Variables", find "Path", click "Edit"
4. Click "New" and add: `C:\Program Files\Tesseract-OCR`
5. Click OK on all dialogs
6. **Restart your terminal/command prompt**

### 5. Alternative: Set Path in Code (Already Implemented)
The vision_agent.py automatically checks these paths:
- `C:\Program Files\Tesseract-OCR\tesseract.exe`
- `C:\Program Files (x86)\Tesseract-OCR\tesseract.exe`
- Environment variable `TESSERACT_PATH`

### 6. Install pytesseract (Python Package)
Already in requirements.txt, but if needed:
```bash
pip install pytesseract pillow
```

## Testing
Run this Python script to test:
```python
import pytesseract
from PIL import Image

# Test OCR
image = Image.open('test_receipt.jpg')
text = pytesseract.image_to_string(image)
print(text)
```

## Troubleshooting

### "tesseract is not installed or it's not in your PATH"
- Verify PATH includes Tesseract folder
- Restart terminal after PATH changes
- Try setting `TESSERACT_PATH` environment variable

### "FileNotFoundError: [WinError 2]"
- Tesseract.exe not found at expected location
- Set explicit path in .env: `TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe`

### Low OCR Accuracy
- Ensure image quality is good (not blurry, good contrast)
- Tesseract works best with clean, high-resolution images
- For poor quality, Gemini Vision fallback will activate (confidence < 70%)

## Performance Notes
- Tesseract: FREE, ~0.5-2 seconds per receipt
- Gemini Vision: Paid API, ~3-5 seconds, better accuracy for complex/poor quality images
- Hybrid approach: Try Tesseract first, fallback to Gemini if confidence low
