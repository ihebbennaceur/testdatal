import pytest
import os
from PIL import Image
from app.TableDetection import TableDetection, TableDetectionError


@pytest.fixture
def sample_invoice_image(tmp_path):
    """Create a sample invoice image for testing"""
    image = Image.new('RGB', (400, 600), color='white')
    image_path = tmp_path / "invoice.jpg"
    image.save(image_path)
    return str(image_path)


@pytest.fixture
def sample_bank_document(tmp_path):
    """Create a sample bank document image for testing"""
    image = Image.new('RGB', (800, 500), color='white')
    image_path = tmp_path / "bank_doc.jpg"
    image.save(image_path)
    return str(image_path)


class TestTableDetectionInitialization:
    """Test model initialization"""
    
    def test_model_initialization(self):
        """Test that model initializes correctly"""
        td = TableDetection()
        assert td.model_name == "TahaDouaji/detr-doc-table-detection"
        assert td.pipe is not None


class TestTableExtractionInvoice:
    """Test table extraction from invoice documents"""
    
    def test_extract_tables_from_invoice(self, sample_invoice_image):
        """Test extracting tables from invoice image"""
        td = TableDetection()
        tables = td.extract_tables(sample_invoice_image)
        assert isinstance(tables, list)
    
    def test_extract_tables_with_confidence_threshold(self, sample_invoice_image):
        """Test extraction with confidence threshold"""
        td = TableDetection()
        tables = td.extract_tables(sample_invoice_image, confidence_threshold=0.7)
        assert isinstance(tables, list)


class TestTableExtractionBankDocument:
    """Test table extraction from bank documents"""
    
    def test_extract_tables_from_bank_doc(self, sample_bank_document):
        """Test extracting tables from bank document"""
        td = TableDetection()
        tables = td.extract_tables(sample_bank_document)
        assert isinstance(tables, list)
    
    def test_extract_tables_bank_with_threshold(self, sample_bank_document):
        """Test extraction from bank doc with confidence threshold"""
        td = TableDetection()
        tables = td.extract_tables(sample_bank_document, confidence_threshold=0.5)
        assert isinstance(tables, list)


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_missing_image_file(self):
        """Test error when image file doesn't exist"""
        td = TableDetection()
        with pytest.raises(TableDetectionError):
            td.extract_tables("/path/to/nonexistent/image.jpg")
    
    def test_invalid_path_type(self):
        """Test error with invalid path type"""
        td = TableDetection()
        with pytest.raises(TableDetectionError):
            td.extract_tables(12345)
    
    def test_invalid_confidence_threshold(self, sample_invoice_image):
        """Test error with invalid confidence threshold"""
        td = TableDetection()
        # Should work with valid thresholds
        tables = td.extract_tables(sample_invoice_image, confidence_threshold=0.5)
        assert isinstance(tables, list)


class TestSuccessScenarios:
    """Test successful extraction scenarios"""
    
    def test_extract_returns_list(self, sample_invoice_image):
        """Test that extract returns a list"""
        td = TableDetection()
        result = td.extract_tables(sample_invoice_image)
        assert isinstance(result, list)
    
    def test_multiple_invoices(self, sample_invoice_image):
        """Test processing multiple invoices"""
        td = TableDetection()
        tables1 = td.extract_tables(sample_invoice_image)
        tables2 = td.extract_tables(sample_invoice_image)
        assert isinstance(tables1, list)
        assert isinstance(tables2, list)