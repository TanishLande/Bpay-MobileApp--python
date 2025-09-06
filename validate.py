# Script 2: Compare Two Face Vectors for Validation
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

def extract_face_vector(image_path):
    """Extract face vector from an image"""
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        faces = app.get(img)
        if len(faces) == 0:
            return None
        
        return faces[0].normed_embedding
        
    except Exception as e:
        print(f"Error: {e}")
        return None

def calculate_similarity(vector1, vector2):
    """
    Calculate cosine similarity between two face vectors
    
    Args:
        vector1: First face vector
        vector2: Second face vector
    
    Returns:
        Similarity score (0-1, higher means more similar)
    """
    # Cosine similarity
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)

def euclidean_distance(vector1, vector2):
    """Calculate Euclidean distance between vectors"""
    return float(np.linalg.norm(vector1 - vector2))

def validate_faces(image1_path, image2_path, threshold=0.6):
    """
    Compare two face images and validate if they are the same person
    
    Args:
        image1_path: Path to first image
        image2_path: Path to second image
        threshold: Similarity threshold for validation (default 0.6)
    
    Returns:
        Boolean indicating if faces match
    """
    print(f"=== FACE COMPARISON ===")
    print(f"Image 1: {image1_path}")
    print(f"Image 2: {image2_path}")
    print(f"Threshold: {threshold}")
    
    # Extract vectors from both images
    print("\nExtracting vector from Image 1...")
    vector1 = extract_face_vector(image1_path)
    
    if vector1 is None:
        print("✗ No face found in Image 1")
        return False
    
    print(f"✓ Vector 1 extracted: {vector1.shape}")
    print(f"Vector 1: {vector1}")
    
    print("\nExtracting vector from Image 2...")
    vector2 = extract_face_vector(image2_path)
    
    if vector2 is None:
        print("✗ No face found in Image 2")
        return False
    
    print(f"✓ Vector 2 extracted: {vector2.shape}")
    print(f"Vector 2: {vector2}")
    
    # Calculate similarity metrics
    similarity = calculate_similarity(vector1, vector2)
    distance = euclidean_distance(vector1, vector2)
    
    print(f"\n=== COMPARISON RESULTS ===")
    print(f"Cosine Similarity: {similarity:.4f}")
    print(f"Euclidean Distance: {distance:.4f}")
    
    # Validate
    is_same_person = similarity >= threshold
    
    print(f"\n=== VALIDATION ===")
    print(f"Threshold: {threshold}")
    print(f"Similarity: {similarity:.4f}")
    
    if is_same_person:
        print("✅ MATCH - Same person!")
    else:
        print("❌ NO MATCH - Different persons!")
    
    # Additional interpretation
    if similarity >= 0.8:
        confidence = "Very High"
    elif similarity >= 0.7:
        confidence = "High"
    elif similarity >= 0.6:
        confidence = "Medium"
    elif similarity >= 0.4:
        confidence = "Low"
    else:
        confidence = "Very Low"
    
    print(f"Confidence Level: {confidence}")
    
    return is_same_person

# Usage example
if __name__ == "__main__":
    # Replace with your image paths
    image1 = "person1_photo1.jpg"  # First image
    image2 = "person1_photo2.jpg"  # Second image to compare
    
    # Compare faces
    result = validate_faces(image1, image2, threshold=0.6)
    
    print(f"\nFinal Result: {'SAME PERSON' if result else 'DIFFERENT PERSON'}")

# Alternative: Load vectors from saved files
def compare_saved_vectors(vector_file1, vector_file2):
    """Compare vectors saved in text files"""
    try:
        vector1 = np.loadtxt(vector_file1, delimiter=',')
        vector2 = np.loadtxt(vector_file2, delimiter=',')
        
        similarity = calculate_similarity(vector1, vector2)
        
        print(f"Similarity between saved vectors: {similarity:.4f}")
        
        return similarity >= 0.6
        
    except Exception as e:
        print(f"Error loading vectors: {e}")
        return False

# Example usage with saved vectors
# result = compare_saved_vectors("vector1.txt", "vector2.txt")