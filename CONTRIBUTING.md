# Contributing to Medical Image Training Platform

Thank you for your interest in contributing! This project welcomes contributions from the community.

## Development Setup

1. **Clone and setup environment:**
```bash
git clone https://github.com/yourusername/medical-image-training.git
cd medical-image-training
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Copy environment template:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Run tests to verify setup:**
```bash
python run_complete_tests.py
```

## Contributing Guidelines

### Code Style
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Add docstrings for all functions and classes
- Keep functions focused and testable

### Testing
- Write tests for new features
- Ensure all existing tests pass
- Aim for >90% code coverage
- Run `python run_complete_tests.py` before submitting

### Pull Request Process
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes with tests
4. Run the complete test suite
5. Commit with clear, descriptive messages
6. Push to your fork and create a pull request

### Areas for Contribution
- **Data augmentation techniques** for medical images
- **New model architectures** (Vision Transformers, EfficientNets)
- **Distributed training optimizations**
- **Medical imaging utilities** (DICOM processing, etc.)
- **Documentation and examples**
- **Performance optimizations**

## Code Review Process
- All submissions require review
- Maintainers will provide feedback within 48 hours
- Address feedback and update PR
- Once approved, maintainers will merge

## Community
- Be respectful and inclusive
- Help others learn and grow
- Share knowledge and best practices

Thank you for contributing! 🎉
