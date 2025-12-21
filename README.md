# Score Sheet OCR

OCR system for digitizing JPA (Japan Poolplayers Association) billiards tournament score sheets.

## Overview

This project provides automated Optical Character Recognition (OCR) to convert physical or scanned JPA billiards score sheets into digital data. Built with Python and Tesseract OCR.

## Features

- Image preprocessing and enhancement
- OCR text extraction using Tesseract
- Data validation for billiards-specific scoring rules
- Support for common pool games (9-ball, 8-ball, 10-ball, etc.)
- Export to CSV/JSON formats

## Requirements

- Python 3.8+
- Tesseract OCR engine

## Installation

### 1. Install Tesseract OCR

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-jpn
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows:**
Download and install from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Clone the repository

```bash
git clone https://github.com/yourusername/score-sheet-ocr.git
cd score-sheet-ocr
```

### 3. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

```
score-sheet-ocr/
├── src/
│   ├── ocr/              # OCR processing logic
│   ├── preprocessing/    # Image preprocessing
│   └── validation/       # Data validation
├── tests/
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── data/
│   ├── raw/             # Original score sheet images
│   ├── processed/       # Preprocessed images
│   └── results/         # OCR output (CSV/JSON)
├── docs/                # Documentation
├── requirements.txt     # Python dependencies
└── claude.md           # Project context for Claude Code
```

## Usage

(To be implemented)

```bash
# Example usage
python -m src.ocr.process_scoresheet data/raw/scoresheet.jpg
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
flake8 src/ tests/
```

## Roadmap

- [ ] Image preprocessing pipeline
- [ ] Tesseract OCR integration
- [ ] Score validation logic
- [ ] CLI interface
- [ ] Batch processing support
- [ ] Web interface (future)

## Contributing

Contributions are welcome! Please see [claude.md](claude.md) for project context and development guidelines.

## License

(To be determined)

## Acknowledgments

- JPA (Japan Poolplayers Association) for the billiards community
- Tesseract OCR team
