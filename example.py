from app.TableDetection import TableDetection, TableDetectionError
import os

try:
    detector = TableDetection()
    print("[OK] Detector initialized")
    
    # Test on real images
    input_folder = "input_folder"
    if os.path.exists(input_folder):
        images = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        if images:
            print(f"\n[OK] Found {len(images)} image(s) to test:\n")
            for img_file in images:
                img_path = os.path.join(input_folder, img_file)
                print(f"  Testing: {img_file}")
                tables = detector.extract_tables(img_path)
                print(f"    -> Found {len(tables)} table(s)")
                for i, table in enumerate(tables, 1):
                    score = table.get('score', 'N/A')
                    if isinstance(score, float):
                        print(f"      Table {i}: score={score:.4f}")
                    else:
                        print(f"      Table {i}: score={score}")
        else:
            print("  No images found in input_folder")
    else:
        print("  input_folder not found")
        
except TableDetectionError as e:
    print(f"[ERROR] {e}")
