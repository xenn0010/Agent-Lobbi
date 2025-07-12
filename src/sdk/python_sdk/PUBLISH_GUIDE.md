# 🚀 Agent Lobbi SDK - PyPI Publication Guide

This guide walks you through publishing the Agent Lobbi Python SDK to PyPI.

## 📋 Prerequisites

### 1. Create PyPI Accounts
- **PyPI**: https://pypi.org/account/register/
- **TestPyPI**: https://test.pypi.org/account/register/
- Verify both email addresses

### 2. Generate API Tokens
- **PyPI Token**: https://pypi.org/manage/account/token/
- **TestPyPI Token**: https://test.pypi.org/manage/account/token/
- Save both tokens securely

### 3. Install Required Tools
```bash
pip install build twine --upgrade
```

## 🔑 Authentication Setup

### Option A: Environment Variables (Recommended)
```bash
# For TestPyPI
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR_TESTPYPI_TOKEN_HERE

# For PyPI
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR_PYPI_TOKEN_HERE
```

### Option B: .pypirc File
Create `~/.pypirc` with:
```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR_REAL_PYPI_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR_TESTPYPI_TOKEN_HERE
```

## 🚀 Publication Methods

### Method 1: Using the Publication Script (Recommended)
```bash
python publish.py
```

### Method 2: Manual Steps

#### Step 1: Clean Previous Builds
```bash
rm -rf dist/ build/
```

#### Step 2: Build Package
```bash
python -m build
```

#### Step 3: Check Package
```bash
python -m twine check dist/*
```

#### Step 4: Upload to TestPyPI (Testing)
```bash
python -m twine upload --repository testpypi dist/*
```

#### Step 5: Test Installation from TestPyPI
```bash
pip install -i https://test.pypi.org/simple/ agent-lobbi-sdk
```

#### Step 6: Upload to PyPI (Production)
```bash
python -m twine upload dist/*
```

## 🧪 Testing the Package

### Test Local Installation
```bash
pip install -e .
python -c "import python_sdk; print('Success!')"
```

### Test CLI
```bash
agent-lobbi --help
```

### Run Test Suite
```bash
python -m pytest tests/ -v
```

## 📦 Package Information

- **Package Name**: `agent-lobbi-sdk`
- **Version**: `1.0.0`
- **CLI Command**: `agent-lobbi`
- **Python Requirements**: `>=3.8`

## 🔗 Links After Publication

### TestPyPI
- **Package Page**: https://test.pypi.org/project/agent-lobbi-sdk/
- **Installation**: `pip install -i https://test.pypi.org/simple/ agent-lobbi-sdk`

### PyPI
- **Package Page**: https://pypi.org/project/agent-lobbi-sdk/
- **Installation**: `pip install agent-lobbi-sdk`

## 🐛 Troubleshooting

### Common Issues

#### 1. Authentication Error (403 Forbidden)
```
ERROR: HTTPError: 403 Forbidden
```
**Solution**: Check your API token and ensure it's correctly set in environment variables or .pypirc

#### 2. Package Already Exists
```
ERROR: File already exists
```
**Solution**: Increment version number in `pyproject.toml` and rebuild

#### 3. Build Errors
```
ERROR: No module named 'build'
```
**Solution**: Install build tools: `pip install build twine --upgrade`

#### 4. Import Errors
```
ImportError: No module named 'python_sdk'
```
**Solution**: Ensure you're in the correct directory and the package structure is correct

### Debug Commands
```bash
# Check package structure
python -m build --help

# Verbose upload
python -m twine upload --verbose --repository testpypi dist/*

# Check installed package
pip show agent-lobbi-sdk
```

## 📊 Post-Publication Checklist

- [ ] Package appears on PyPI/TestPyPI
- [ ] Installation works: `pip install agent-lobbi-sdk`
- [ ] CLI works: `agent-lobbi --help`
- [ ] Import works: `python -c "import python_sdk"`
- [ ] Documentation is accessible
- [ ] Examples run successfully

## 🔄 Version Updates

For future updates:
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Rebuild and republish
4. Tag the release in Git

## 🆘 Support

If you encounter issues:
- Check the [PyPI Documentation](https://packaging.python.org/tutorials/packaging-projects/)
- Review [Twine Documentation](https://twine.readthedocs.io/)
- Open an issue in the repository

---

**Happy Publishing! 🎉** 