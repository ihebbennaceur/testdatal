# Table Detection

Extract tables from invoice and bank documents using DETR transformer model.

## Setup

```bash
pip install -r requirements.txt
```

## Run Tests

```bash
pytest tests/
```

## Usage

```python
from app.TableDetection import TableDetection

detector = TableDetection()
tables = detector.extract_tables("invoice.jpg")
```

## Project Structure

- `app/TableDetection.py` - Main class
- `tests/test_invoice.py` - Tests (10 test cases)
- `requirements.txt` - Dependencies

## Model

Uses pre-trained DETR model: https://huggingface.co/TahaDouaji/detr-doc-table-detection
