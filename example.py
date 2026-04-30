from app.TableDetection import TableDetection, TableDetectionError

try:
    detector = TableDetection()
    print("Detector initialized")
except TableDetectionError as e:
    print(f"Error: {e}")
