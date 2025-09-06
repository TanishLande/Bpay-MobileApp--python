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
        print("Files in current directory:")
        for file in os.listdir('.'):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"  - {file}")
        return False
    return True

def extract_face_vector_with_details(image_path):
    """Extract face vector with detailed output"""
    
    if not check_image_exists(image_path):
        return None
    
    try:
        print(f"\n=== PROCESSING IMAGE: {image_path} ===")
        
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
        
        print(f"\n=== FACE VECTOR EXTRACTED ===")
        print(f"Vector shape: {vector.shape}")
        print(f"Vector type: {type(vector)}")
        print(f"Vector dtype: {vector.dtype}")
        print(f"\n📊 FACE VECTOR DATA:")
        print(f"Vector: {vector}")
        
        # Save vector to file
        vector_filename = f"face_vector_{os.path.splitext(os.path.basename(image_path))[0]}.txt"
        np.savetxt(vector_filename, vector, delimiter=',', fmt='%.8f')
        print(f"✓ Vector saved to: {vector_filename}")
        
        # Display some statistics
        print(f"\n📈 VECTOR STATISTICS:")
        print(f"Min value: {np.min(vector):.6f}")
        print(f"Max value: {np.max(vector):.6f}")
        print(f"Mean value: {np.mean(vector):.6f}")
        print(f"Std deviation: {np.std(vector):.6f}")
        
        return vector
        
    except Exception as e:
        print(f"✗ Error during face extraction: {e}")
        return None

def calculate_similarity(vector1, vector2):
    """Calculate cosine similarity between two face vectors"""
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)

def validate_with_another_image(image1_path, image2_path, threshold=0.6):
    """Compare two images and validate if they show the same person"""
    
    print(f"\n{'='*50}")
    print(f"🔍 FACE VALIDATION COMPARISON")
    print(f"{'='*50}")
    print(f"Image 1: {image1_path}")
    print(f"Image 2: {image2_path}")
    print(f"Similarity Threshold: {threshold}")
    
    # Extract vector from first image
    print(f"\n--- PROCESSING FIRST IMAGE ---")
    vector1 = extract_face_vector_with_details(image1_path)
    
    if vector1 is None:
        print("❌ Cannot proceed - failed to extract vector from first image")
        return False
    
    # Extract vector from second image
    print(f"\n--- PROCESSING SECOND IMAGE ---")
    vector2 = extract_face_vector_with_details(image2_path)
    
    if vector2 is None:
        print("❌ Cannot proceed - failed to extract vector from second image")
        return False
    
    # Calculate similarity
    similarity = calculate_similarity(vector1, vector2)
    distance = float(np.linalg.norm(vector1 - vector2))
    
    print(f"\n{'='*50}")
    print(f"📊 COMPARISON RESULTS")
    print(f"{'='*50}")
    print(f"Cosine Similarity: {similarity:.6f}")
    print(f"Euclidean Distance: {distance:.6f}")
    print(f"Threshold: {threshold}")
    
    # Determine if same person
    is_same_person = similarity >= threshold
    
    # Confidence levels
    if similarity >= 0.9:
        confidence = "Extremely High"
        emoji = "🟢"
    elif similarity >= 0.8:
        confidence = "Very High"
        emoji = "🟢"
    elif similarity >= 0.7:
        confidence = "High"
        emoji = "🟡"
    elif similarity >= 0.6:
        confidence = "Medium"
        emoji = "🟡"
    elif similarity >= 0.4:
        confidence = "Low"
        emoji = "🟠"
    else:
        confidence = "Very Low"
        emoji = "🔴"
    
    print(f"\n🎯 VALIDATION RESULT:")
    if is_same_person:
        print(f"✅ MATCH - Same person detected!")
        print(f"{emoji} Confidence: {confidence} ({similarity:.3f})")
    else:
        print(f"❌ NO MATCH - Different persons detected!")
        print(f"{emoji} Confidence: {confidence} ({similarity:.3f})")
    
    return is_same_person, similarity

def main():
    """Main execution function"""
    
    # Network diagnostics
    print("=== Network Diagnostics ===")
    try:
        import socket
        socket.gethostbyname('github.com')
        print("✓ DNS resolution working")
    except:
        print("✗ DNS resolution failed")
    
    print("\n" + "="*60)
    print("🔥 FACE RECOGNITION SYSTEM")
    print("="*60)
    
    # Download model if needed
    model_dir = Path.home() / '.insightface' / 'models' / 'buffalo_l'
    if not model_dir.exists():
        print("Model not found. Downloading...")
        if not download_model_manually():
            print("❌ Failed to download model. Exiting.")
            return
    else:
        print("✓ Model already available")
    
    # Configuration - UPDATE THESE PATHS
    image1_path = "vk.png"  # Your first image
    image2_path = "rohit.png"  # Your second image for comparison (optional)
    
    # Extract vector from primary image
    print(f"\n🎯 EXTRACTING FACE VECTOR FROM PRIMARY IMAGE")
    vector1 = extract_face_vector_with_details(image1_path)
    
    if vector1 is None:
        print("❌ Failed to extract face vector from primary image")
        return
    
    # Check if second image exists for validation
    if os.path.exists(image2_path):
        print(f"\n🔍 SECOND IMAGE FOUND - PERFORMING VALIDATION")
        result, similarity = validate_with_another_image(image1_path, image2_path)
        
        print(f"\n{'='*60}")
        print(f"🏁 FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Primary Image: {image1_path}")
        print(f"Comparison Image: {image2_path}")
        print(f"Similarity Score: {similarity:.6f}")
        print(f"Result: {'SAME PERSON ✅' if result else 'DIFFERENT PERSON ❌'}")
        
    else:
        print(f"\n💡 Second image '{image2_path}' not found.")
        print(f"To perform validation, add a second image file named '{image2_path}'")
        print(f"Primary image vector has been successfully extracted and saved!")

if __name__ == "__main__":
    main()