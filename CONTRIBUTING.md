# Contributing to MuSc Industrial GUI

Thank you for your interest in contributing to MuSc Industrial GUI! This document provides guidelines and instructions for contributing.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [What NOT to Contribute](#what-not-to-contribute)

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/musc-gui.git
   cd musc-gui/MuSc
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/musc-gui.git
   ```

## Development Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Git

### Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

2. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks** (recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

4. **Verify installation**:
   ```bash
   # Run tests
   pytest tests/

   # Launch GUI
   musc-gui
   ```

## Making Changes

### Branch Naming
- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Workflow
1. **Create a branch** from `main`:
   ```bash
   git checkout main
   git pull upstream main
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Test your changes**:
   ```bash
   pytest tests/
   ```

4. **Commit with descriptive messages**:
   ```bash
   git add .
   git commit -m "Add feature: brief description"
   ```

## Code Style

### Python Style
- Follow PEP 8 guidelines
- Use meaningful variable names
- Keep functions focused and reasonably sized
- Add type hints for function signatures

### Formatting
We use `black` and `ruff` for code formatting. Run before committing:
```bash
black .
ruff check --fix .
```

### Documentation
- Add docstrings to public functions and classes
- Update README.md if adding user-facing features
- Keep comments concise and meaningful

### Example Function
```python
def process_image(
    image: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, float]:
    """
    Process an image for anomaly detection.

    Args:
        image: Input image as numpy array (H, W, C).
        threshold: Detection threshold between 0 and 1.

    Returns:
        Tuple of (anomaly_map, anomaly_score).
    """
    # Implementation
    pass
```

## Testing

### Running Tests
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=models --cov-report=html

# Run specific test file
pytest tests/test_smoke.py

# Run with verbose output
pytest tests/ -v
```

### Writing Tests
- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names
- Test both success and failure cases

```python
def test_musc_inference_with_valid_image():
    """Test that inference works with a valid input image."""
    # Test implementation
    pass

def test_musc_inference_with_invalid_input():
    """Test that inference raises appropriate error for invalid input."""
    # Test implementation
    pass
```

## Submitting Changes

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** on GitHub:
   - Fill out the PR template completely
   - Link any related issues
   - Add screenshots for UI changes

3. **Respond to feedback**:
   - Address reviewer comments
   - Push additional commits as needed

4. **Merge**: Once approved, your PR will be merged

## What NOT to Contribute

To keep the project focused, please avoid:

- **Core algorithm changes**: The MuSc algorithm (LNAMD, MSM, RsCIN) is stable and should not be modified without discussion
- **New backbone models**: We already support 14+ backbones; additional ones need strong justification
- **Web interface**: This is a desktop GUI project; web interfaces should be separate projects
- **Training/fine-tuning**: Zero-shot detection is the key differentiator; supervised approaches don't fit
- **Breaking changes**: Avoid changes that break existing functionality without discussion

### Good Contributions
- Bug fixes
- Documentation improvements
- Test coverage
- Performance optimizations
- UI/UX improvements
- Export format additions
- Platform compatibility fixes

## Questions?

- Open an issue for questions or discussion
- Check existing issues before creating new ones

Thank you for contributing!
