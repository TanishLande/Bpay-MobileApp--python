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
        app = FaceAnalysis(providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"✗ Error: Could not read image from {image_path}")
            return None
        
        print(f"✓ Image loaded successfully! Shape: {img.shape}")
        
        # Detect faces
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
        
        return vector
        
    except Exception as e:
        print(f"✗ Error during face extraction: {e}")
        return None

def calculate_similarity(vector1, vector2):
    """
    Calculate cosine similarity between two face vectors
    
    Args:
        vector1 (numpy.ndarray): First face vector
        vector2 (numpy.ndarray): Second face vector
    
    Returns:
        float: Cosine similarity score between -1 and 1
    """
    # Convert lists back to numpy arrays if needed
    if isinstance(vector1, list):
        vector1 = np.array(vector1)
    if isinstance(vector2, list):
        vector2 = np.array(vector2)
    
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)

def compare_faces(image1_path, image2_path, threshold=0.6):
    """
    Compare two face images and return True/False if they match
    
    Args:
        image1_path (str): Path to first image
        image2_path (str): Path to second image
        threshold (float): Similarity threshold (default: 0.6)
    
    Returns:
        bool: True if faces match above threshold, False otherwise
        float: Similarity score
    """
    
    print(f"\n{'='*60}")
    print(f"🔍 COMPARING TWO FACES")
    print(f"{'='*60}")
    print(f"Image 1: {image1_path}")
    print(f"Image 2: {image2_path}")
    print(f"Threshold: {threshold}")
    
    # Extract vector from first image
    print(f"\n--- PROCESSING FIRST IMAGE ---")
    vector1 = extract_face_vector(image1_path)
    
    if vector1 is None:
        print("❌ Failed to extract vector from first image")
        return False, 0.0
    
    # Extract vector from second image
    print(f"\n--- PROCESSING SECOND IMAGE ---")
    vector2 = extract_face_vector(image2_path)
    
    if vector2 is None:
        print("❌ Failed to extract vector from second image")
        return False, 0.0
    
    # Calculate similarity
    similarity = calculate_similarity(vector1, vector2)
    
    print(f"\n{'='*60}")
    print(f"📊 COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Cosine Similarity: {similarity:.6f}")
    print(f"Threshold: {threshold}")
    
    # Determine if same person
    is_match = similarity >= threshold
    
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
    
    print(f"\n🎯 COMPARISON RESULT:")
    if is_match:
        print(f"✅ MATCH - Same person detected!")
        print(f"{emoji} Confidence: {confidence} ({similarity:.3f})")
    else:
        print(f"❌ NO MATCH - Different persons detected!")
        print(f"{emoji} Confidence: {confidence} ({similarity:.3f})")
    
    return is_match, similarity

def compare_face_vectors(vector1, vector2, threshold=0.6):
    """
    Compare two face vectors directly and return True/False if they match
    
    Args:
        vector1 (numpy.ndarray): First face vector
        vector2 (numpy.ndarray): Second face vector  
        threshold (float): Similarity threshold (default: 0.6)
    
    Returns:
        bool: True if vectors match above threshold, False otherwise
        float: Similarity score
    """
    
    print(f"\n{'='*60}")
    print(f"🔍 COMPARING TWO FACE VECTORS")
    print(f"{'='*60}")
    print(f"Threshold: {threshold}")
    
    if vector1 is None or vector2 is None:
        print("❌ One or both vectors are None")
        return False, 0.0
    
    # Calculate similarity
    similarity = calculate_similarity(vector1, vector2)
    
    print(f"Cosine Similarity: {similarity:.6f}")
    
    # Determine if same person
    is_match = similarity >= threshold
    
    print(f"\n🎯 VECTOR COMPARISON RESULT:")
    if is_match:
        print(f"✅ MATCH - Vectors match above threshold!")
        print(f"Similarity: {similarity:.6f}")
    else:
        print(f"❌ NO MATCH - Vectors don't match!")
        print(f"Similarity: {similarity:.6f}")
    
    return is_match, similarity

def main():
    """Main execution function for testing"""
    
    # Network diagnostics
    print("=== Network Diagnostics ===")
    try:
        import socket
        socket.gethostbyname('github.com')
        print("✓ DNS resolution working")
    except:
        print("✗ DNS resolution failed")
    
    print("\n" + "="*60)
    print("🔥 FACE RECOGNITION - VECTOR EXTRACTION & COMPARISON")
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
    
    # Test with sample images
    image1 = "vk.png"    # Change to your first image
    image2 = "rohit.png" # Change to your second image
    
    # Example 1: Extract individual vectors
    if os.path.exists(image1):
        print(f"\n🔥 EXTRACTING VECTOR FROM: {image1}")
        vector1 = extract_face_vector(image1)
        
        if vector1 is not None:
            print(f"✅ Successfully extracted vector from {image1}")
    
    if os.path.exists(image2):
        print(f"\n🔥 EXTRACTING VECTOR FROM: {image2}")
        vector2 = extract_face_vector(image2)
        
        if vector2 is not None:
            print(f"✅ Successfully extracted vector from {image2}")
    
    # Example 2: Compare two images directly
    if os.path.exists(image1) and os.path.exists(image2):
        print(f"\n🔥 COMPARING IMAGES DIRECTLY")
        is_match, similarity = compare_faces(image1, image2, threshold=0.6)
        
        print(f"\n📋 FINAL RESULT:")
        print(f"Images match: {is_match}")
        print(f"Similarity score: {similarity:.6f}")
    
    # Example 3: Compare same image with itself (should be very high similarity)
    if os.path.exists(image1):
        print(f"\n🔥 COMPARING SAME IMAGE WITH ITSELF (SHOULD MATCH)")
        is_match, similarity = compare_faces(image1, image1, threshold=0.6)
        
        print(f"\n📋 SELF-COMPARISON RESULT:")
        print(f"Images match: {is_match}")
        print(f"Similarity score: {similarity:.6f}")

if __name__ == "__main__":
    main()