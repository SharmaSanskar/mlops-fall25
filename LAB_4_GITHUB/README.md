# GitHub Lab

## Additional Implementations

- **Text Analyzer Tool** instead of calculator - 6 functions for word count, character count, sentence analysis, word frequency, average word length, and comprehensive analysis
- **20+ test cases** covering edge cases, parametrized tests, and error handling
- **Regular expressions** for sentence splitting and punctuation removal
- **Python Counter** for efficient word frequency analysis
- **Path filters** in GitHub Actions to only trigger LAB_4 workflows when LAB_4_GITHUB files change

---

## Screenshots
<img width="953" height="360" alt="Screenshot 2025-11-03 at 7 03 41 PM" src="https://github.com/user-attachments/assets/139ec4da-3b8c-4968-9a6f-f2754772dbbe" />

<img width="940" height="359" alt="Screenshot 2025-11-03 at 7 04 42 PM" src="https://github.com/user-attachments/assets/e9e0c860-afdc-44c6-82c5-c0c38f9efcf2" />

<img width="840" height="402" alt="Screenshot 2025-11-03 at 7 04 30 PM" src="https://github.com/user-attachments/assets/14a45329-b227-4230-a750-6c0ad9e5b955" />

---

## Step 1: Creating a Virtual Environment
```bash
cd ~/Documents/mlops-fall25
mkdir LAB_4_GITHUB
cd LAB_4_GITHUB
python -m venv venv
```

Activate:
- Mac/Linux: `source venv/bin/activate`
- Windows: `venv\Scripts\activate`

---

## Step 2: Folder Structure
```bash
mkdir src test data
```

Created `.gitignore` with:
```
venv/
__pycache__/
*.pyc
.pytest_cache/
*.xml
```

---

## Step 3: Creating text_analyzer.py

Created `src/text_analyzer.py` with functions:
- `count_words(text)` - Count words
- `count_characters(text, include_spaces)` - Count characters
- `count_sentences(text)` - Count sentences using regex
- `get_word_frequency(text, top_n)` - Most common words
- `calculate_average_word_length(text)` - Average word length
- `analyze_text(text)` - All metrics combined

---

## Step 4: Creating Tests

### Pytest Tests
Created `test/test_pytest.py` with 10+ test functions.

Run with:
```bash
pytest test/test_pytest.py -v
```

### Unittest Tests
Created `test/test_unittest.py` with 10+ test methods.

Run with:
```bash
python -m unittest test.test_unittest -v
```

---

## Step 5: GitHub Actions

Created two workflow files in `.github/workflows/`:

### lab4_pytest_action.yml
- Triggers on push/PR to main when LAB_4_GITHUB files change
- Runs pytest tests
- Uploads test results as artifacts
- Uses actions v4/v5

### lab4_unittest_action.yml
- Triggers on push/PR to main when LAB_4_GITHUB files change
- Runs unittest tests
- Shows success/failure notifications
- Uses actions v4/v5

---

## Step 6: Push to GitHub
```bash
git add .
git commit -m "Add LAB_4_GITHUB: Text analyzer with CI/CD"
git push origin main
```

---

## Requirements

`requirements.txt`:
```
pytest>=7.0.0
pytest-cov>=4.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

---
