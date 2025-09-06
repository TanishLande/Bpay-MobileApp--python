import os
import requests
from pathlib import Path
import zipfile
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

def download_model_manually():
    """Manually download and setup InsightFace model"""
    model_dir = Path.home() / '.insightface' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    model_path = model_dir / "buffalo_l.zip"
    
    print(f"Downloading model to: {model_path}")
    
    try:
        response = requests.get(model_url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rDownload progress: {progress:.1f}%", end='', flush=True)
        
        print("\n✓ Download completed!")
        
        # Extract the zip file
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        
        print("✓ Model extracted successfully!")
        model_path.unlink()  # Remove zip file
        
        return True
        
    except Exception as e:
        print(f"✗ Manual download failed: {e}")
        return False

def check_image_exists(image_path):
    """Check if image file exists and is readable"""
    if not os.path.exists(image_path):
        print(f"✗ Error: Image file '{image_path}' not found!")
        print(f"Current directory: {os.getcwd()}")
        print("Available image files:")
        for file in os.listdir('.'):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"  - {file}")
        return False
    return True

def extract_face_vector(image_path):
    """
    Extract face vector from image
    
    Args:
        image_path (str): Path to the image file
    
    Returns:
        numpy.ndarray or None: Face vector if successful, None if failed
    """
    
    if not check_image_exists(image_path):
        return None
    
    try:
        # Initialize InsightFace
        print("Initializing InsightFace...")
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("✓ InsightFace initialized successfully!")
        
        # Read image
        print(f"Reading image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"✗ Error: Could not read image from {image_path}")
            return None
        
        print(f"✓ Image loaded successfully! Shape: {img.shape}")
        
        # Detect faces
        print("Detecting faces...")
        faces = app.get(img)
        
        if len(faces) == 0:
            print("✗ No face detected in the image!")
            return None
        
        print(f"✓ {len(faces)} face(s) detected!")
        
        # Extract vector from first face
        face = faces[0]
        vector = face.normed_embedding
        
        print(f"✓ Face vector extracted successfully!")
        print(f"Vector shape: {vector.shape}")
        print(f"Vector type: {type(vector)}")
        
        # Print the vector
        print(f"\n📊 EXTRACTED FACE VECTOR:")
        print("="*40)
        print(f"Vector: {vector}")
        print("="*40)
        
        # Display vector statistics
        print(f"\n📈 VECTOR STATISTICS:")
        print(f"Min value: {np.min(vector):.6f}")
        print(f"Max value: {np.max(vector):.6f}")
        print(f"Mean value: {np.mean(vector):.6f}")
        print(f"Std deviation: {np.std(vector):.6f}")
        
        # Save vector to file
        # vector_filename = f"face_vector_{os.path.splitext(os.path.basename(image_path))[0]}.txt"
        # np.savetxt(vector_filename, vector, delimiter=',', fmt='%.8f')
        # print(f"✓ Vector saved to: {vector_filename}")
        
        return vector
        
    except Exception as e:
        print(f"✗ Error during face extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to run the face extraction"""
    
    print("="*60)
    print("🔥 FACE VECTOR EXTRACTION SYSTEM")
    print("="*60)
    
    # Check network connectivity
    try:
        import socket
        socket.gethostbyname('github.com')
        print("✓ Network connection available")
    except:
        print("⚠️ Network connection issues detected")
    
    # Check and download model if needed
    model_dir = Path.home() / '.insightface' / 'models' / 'buffalo_l'
    if not model_dir.exists():
        print("\n📥 Model not found. Downloading...")
        if not download_model_manually():
            print("❌ Failed to download model. Please check your internet connection.")
            return
    else:
        print("✓ InsightFace model already available")
    
    # Test images - update these paths to your actual image files
    test_images = [
        "vk.png"
    ]
    
    
    # Process first available image
    test_image = test_images[0]
    print(f"\n🎯 Processing: {test_image}")
    
    vector = extract_face_vector(test_image)
    
    if vector is not None:
        print(f"\n✅ SUCCESS! Face vector extracted from {test_image}")
        print(f"Vector dimension: {vector.shape[0]}")
        print(f"Vector saved to file for future use.")
    else:
        print(f"\n❌ FAILED to extract face vector from {test_image}")
        print("Make sure the image contains a clear, visible face.")


if __name__ == "__main__":
    main()