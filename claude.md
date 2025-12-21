# Claude Code Reference - Score Sheet OCR Project

## Project Overview
This project provides OCR (Optical Character Recognition) functionality for JPA (Japan Poolplayers Association) score sheets.

## Project Context
- **Purpose**: Digitize and process JPA (Japan Poolplayers Association) billiards tournament score sheets using OCR technology
- **Sport**: Billiards/Pool
- **Type**: OCR/Image Processing Application
- **Primary Language**: Python 3.8+
- **OCR Engine**: Tesseract OCR (with Japanese language support)
- **Main Branch**: main

## Development Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use Black for code formatting
- Use Flake8 for linting
- Use clear, descriptive variable and function names
- Add docstrings to all functions and classes
- Add comments for complex logic
- Keep functions focused and modular
- Type hints encouraged for better code clarity

### Git Workflow
- Main branch: `main`
- Create feature branches for new functionality
- Write clear, descriptive commit messages
- Use conventional commits format when possible (feat:, fix:, docs:, etc.)

### File Organization
```
score-sheet-ocr/
├── src/
│   ├── __init__.py
│   ├── ocr/              # OCR processing logic
│   │   └── __init__.py
│   ├── preprocessing/    # Image preprocessing
│   │   └── __init__.py
│   └── validation/       # Data validation
│       └── __init__.py
├── tests/
│   ├── __init__.py
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
├── data/
│   ├── raw/             # Original score sheet images
│   ├── processed/       # Preprocessed images
│   └── results/         # OCR output (CSV/JSON)
├── docs/                # Documentation
├── .gitignore           # Git ignore rules
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── claude.md            # This file - Claude Code reference
```

## Key Features (To Be Implemented)
- OCR processing for JPA score sheets
- Image preprocessing and enhancement
- Data extraction and validation
- Export functionality for digitized scores

## Common Tasks

### Development
- Test OCR accuracy with sample score sheets
- Validate extracted data against known values
- Handle edge cases (poor image quality, handwriting variations)

### Dependencies
- **OCR**: pytesseract (Python wrapper for Tesseract OCR)
- **Image Processing**: Pillow, OpenCV (cv2), NumPy
- **PDF Support**: pdf2image (optional, for PDF score sheets)
- **Data Processing**: pandas (for CSV/JSON export)
- **Testing**: pytest, pytest-cov
- **Code Quality**: black (formatter), flake8 (linter)

## Important Notes

### Billiards Context
- JPA: Japan Poolplayers Association - billiards/pool organization
- Score sheets track match results, player names, game scores, and tournament information
- Common pool games: 9-ball, 8-ball, 10-ball, rotation, etc.
- Score formats vary by game type and tournament rules

### OCR Considerations
- Score sheets may have handwritten entries (player signatures, scores)
- Image quality varies - implement preprocessing
- Consider multiple OCR engines for better accuracy
- Implement confidence scoring for extracted data
- Handle billiards-specific terminology and abbreviations

### Data Validation
- Verify score ranges are valid for specific game types
- Check for completeness of required fields (player names, dates, scores)
- Validate game scores against billiards rules (e.g., 9-ball games typically go to 9 or race format)
- Flag low-confidence OCR results for manual review
- Detect and handle common OCR errors (e.g., "0" vs "O", "1" vs "I")

## Testing Strategy
- Unit tests for individual components
- Integration tests for OCR pipeline
- Accuracy tests with known sample sheets
- Performance benchmarks for processing speed

## Security & Privacy
- Handle score data appropriately
- Consider data retention policies
- Implement access controls if storing results
- Sanitize file uploads to prevent malicious files

## Resources
- JPA (Japan Poolplayers Association): Billiards organization in Japan
- JPA Score Sheet Format: [To be documented - varies by tournament/game type]
- OCR Best Practices: [To be referenced]
- Image Processing Guidelines: [To be referenced]
- Billiards terminology and scoring rules reference: [To be added]

## Technical Stack (MVP)
- **Language**: Python 3.8+
- **OCR Engine**: Tesseract OCR with Japanese language support
- **Image Processing**: OpenCV, Pillow
- **Interface**: CLI (Command Line Interface) for MVP
- **Output Formats**: CSV, JSON
- **Processing**: Batch processing support

## Implementation Phases

### Phase 1: MVP (Current)
- [x] Project setup and structure
- [ ] Basic image preprocessing (grayscale, threshold, noise removal)
- [ ] Tesseract OCR integration
- [ ] Simple CLI for single image processing
- [ ] Basic data extraction (text-only)
- [ ] CSV/JSON export

### Phase 2: Enhancement
- [ ] Advanced preprocessing (deskew, perspective correction)
- [ ] Template-based field detection
- [ ] Data validation against billiards rules
- [ ] Batch processing multiple images
- [ ] Confidence scoring and manual review flags
- [ ] Error handling and logging

### Phase 3: Future
- [ ] Web interface
- [ ] Database storage
- [ ] Multiple OCR engine support
- [ ] Machine learning for field detection
- [ ] API for integration with other systems

## Next Steps
- Document JPA score sheet format and fields
- Implement image preprocessing pipeline
- Set up Tesseract OCR integration
- Create data validation rules
- Build CLI interface
- Implement testing suite
- Write user documentation

---

**Last Updated**: 2025-12-21
**Project Status**: Initial Setup
