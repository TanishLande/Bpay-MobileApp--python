import os
import requests
from pathlib import Path
import zipfile
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import json
from datetime import datetime

# In-memory database simulation (replace with actual database logic later)
FACE_DATABASE = {}

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
    """Extract face vector from image"""
    
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
        
        # Detect faces
        faces = app.get(img)
        
        if len(faces) == 0:
            print("✗ No face detected in the image!")
            return None
        
        # Extract vector from first face
        face = faces[0]
        vector = face.normed_embedding
        
        return vector
        
    except Exception as e:
        print(f"✗ Error during face extraction: {e}")
        return None

def store_face_in_database(image_path, person_info=None):
    """
    PART 1: Store face vector in database
    Convert image to vector and store with associated information
    """
    
    print(f"\n{'='*60}")
    print(f"📥 STORING FACE IN DATABASE")
    print(f"{'='*60}")
    print(f"Processing image: {image_path}")
    
    # Extract face vector
    vector = extract_face_vector(image_path)
    
    if vector is None:
        print("❌ Failed to extract face vector. Cannot store in database.")
        return False
    
    print(f"✓ Face vector extracted successfully!")
    print(f"Vector shape: {vector.shape}")
    print(f"Vector type: {type(vector)}")
    
    # Print the vector (as requested - no database logic, just print)
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
    
    # Simulate database storage (replace with actual database logic)
    person_id = len(FACE_DATABASE) + 1
    
    # Default person info if not provided
    if person_info is None:
        person_info = {
            "name": f"Person_{person_id}",
            "image_path": image_path,
            "age": "Unknown",
            "department": "Unknown",
            "employee_id": f"EMP_{person_id:03d}",
            "phone": "Not provided",
            "email": "Not provided"
        }
    
    # Store in simulated database
    FACE_DATABASE[person_id] = {
        "person_id": person_id,
        "vector": vector.tolist(),  # Convert numpy array to list for JSON serialization
        "info": person_info,
        "created_at": datetime.now().isoformat(),
        "image_path": image_path
    }
    
    print(f"\n✅ FACE STORED IN DATABASE")
    print(f"Person ID: {person_id}")
    print(f"Associated Info: {json.dumps(person_info, indent=2)}")
    
    # Save vector to file for backup
    vector_filename = f"stored_vector_{person_id}_{os.path.splitext(os.path.basename(image_path))[0]}.txt"
    np.savetxt(vector_filename, vector, delimiter=',', fmt='%.8f')
    print(f"✓ Vector backup saved to: {vector_filename}")
    
    return True

def calculate_similarity(vector1, vector2):
    """Calculate cosine similarity between two face vectors"""
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

def search_face_in_database(input_image_path, similarity_threshold=0.6):
    """
    PART 2: Search for matching face in database
    Convert input image to vector and search for matching vectors in database
    """
    
    print(f"\n{'='*60}")
    print(f"🔍 SEARCHING FACE IN DATABASE")
    print(f"{'='*60}")
    print(f"Input image: {input_image_path}")
    print(f"Similarity threshold: {similarity_threshold}")
    
    # Extract face vector from input image
    print(f"\nExtracting face vector from input image...")
    input_vector = extract_face_vector(input_image_path)
    
    if input_vector is None:
        print("❌ Failed to extract face vector from input image. Cannot search.")
        return None
    
    print(f"✓ Input face vector extracted successfully!")
    print(f"Vector shape: {input_vector.shape}")
    
    # Print input vector
    print(f"\n📊 INPUT FACE VECTOR:")
    print("="*40)
    print(f"Vector: {input_vector}")
    print("="*40)
    
    # Check if database is empty
    if not FACE_DATABASE:
        print("\n❌ Database is empty! No faces to search against.")
        print("Please store some faces first using store_face_in_database()")
        return None
    
    print(f"\n🔍 Searching against {len(FACE_DATABASE)} stored faces...")
    
    # Search for matches
    matches = []
    
    for person_id, stored_data in FACE_DATABASE.items():
        stored_vector = np.array(stored_data["vector"])
        similarity = calculate_similarity(input_vector, stored_vector)
        
        if similarity >= similarity_threshold:
            matches.append({
                "person_id": person_id,
                "similarity": similarity,
                "info": stored_data["info"],
                "stored_image": stored_data["image_path"],
                "created_at": stored_data["created_at"]
            })
    
    # Sort matches by similarity (highest first)
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"📊 SEARCH RESULTS")
    print(f"{'='*60}")
    
    if not matches:
        print("❌ NO MATCHES FOUND")
        print(f"No faces in database match with similarity >= {similarity_threshold}")
        
        # Show closest matches anyway
        print(f"\n🔍 CLOSEST MATCHES (below threshold):")
        all_similarities = []
        for person_id, stored_data in FACE_DATABASE.items():
            stored_vector = np.array(stored_data["vector"])
            similarity = calculate_similarity(input_vector, stored_vector)
            all_similarities.append({
                "person_id": person_id,
                "similarity": similarity,
                "info": stored_data["info"]
            })
        
        all_similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        for i, match in enumerate(all_similarities[:3], 1):  # Show top 3
            print(f"\n{i}. Person ID: {match['person_id']}")
            print(f"   Similarity: {match['similarity']:.6f}")
            print(f"   Name: {match['info']['name']}")
    
    else:
        print(f"✅ FOUND {len(matches)} MATCHING FACE(S)")
        
        for i, match in enumerate(matches, 1):
            confidence_emoji = "🟢" if match['similarity'] >= 0.8 else "🟡" if match['similarity'] >= 0.7 else "🟠"
            
            print(f"\n{confidence_emoji} MATCH #{i}")
            print(f"{'='*30}")
            print(f"Person ID: {match['person_id']}")
            print(f"Similarity Score: {match['similarity']:.6f}")
            print(f"Stored Image: {match['stored_image']}")
            print(f"Created: {match['created_at']}")
            print(f"\n👤 PERSON INFORMATION:")
            for key, value in match['info'].items():
                print(f"  {key.title()}: {value}")
    
    return matches

def display_database_contents():
    """Display all stored faces in database"""
    
    print(f"\n{'='*60}")
    print(f"💾 DATABASE CONTENTS")
    print(f"{'='*60}")
    
    if not FACE_DATABASE:
        print("Database is empty!")
        return
    
    print(f"Total stored faces: {len(FACE_DATABASE)}")
    
    for person_id, data in FACE_DATABASE.items():
        print(f"\n--- Person ID: {person_id} ---")
        print(f"Image: {data['image_path']}")
        print(f"Created: {data['created_at']}")
        print(f"Info: {json.dumps(data['info'], indent=2)}")

def main():
    """Main execution function with options"""
    
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
    
    # DEMO: Store some faces first
    print(f"\n🔥 DEMO: STORING FACES IN DATABASE")
    
    # Example person info (customize as needed)
    person1_info = {
        "name": "Virat Kohli",
        "age": "35",
        "department": "Cricket Team",
        "employee_id": "CRIC_001",
        "phone": "+91-9876543210",
        "email": "virat@cricket.com"
    }
    
    person2_info = {
        "name": "Rohit Sharma",
        "age": "37",
        "department": "Cricket Team",
        "employee_id": "CRIC_002",
        "phone": "+91-9876543211",
        "email": "rohit@cricket.com"
    }
    
    # PART 1: Store faces (update image paths as needed)
    if os.path.exists("vk.png"):
        store_face_in_database("vk.png", person1_info)
    
    if os.path.exists("rohit.png"):
        store_face_in_database("rohit.png", person2_info)
    
    # Display database contents
    display_database_contents()
    
    # PART 2: Search for a face (update search image path as needed)
    search_image = "vk.png"  # Change this to your search image
    
    if os.path.exists(search_image):
        print(f"\n🔥 DEMO: SEARCHING FOR FACE")
        matches = search_face_in_database(search_image, similarity_threshold=0.6)
    else:
        print(f"\n💡 Search image '{search_image}' not found.")
        print("Add an image file to test the search functionality.")

# Additional utility functions for standalone usage

def store_single_face(image_path, person_name="Unknown", **kwargs):
    """Simplified function to store a single face"""
    person_info = {
        "name": person_name,
        "age": kwargs.get("age", "Unknown"),
        "department": kwargs.get("department", "Unknown"),
        "employee_id": kwargs.get("employee_id", f"EMP_{len(FACE_DATABASE) + 1:03d}"),
        "phone": kwargs.get("phone", "Not provided"),
        "email": kwargs.get("email", "Not provided")
    }
    return store_face_in_database(image_path, person_info)

def search_single_face(image_path, threshold=0.6):
    """Simplified function to search for a single face"""
    return search_face_in_database(image_path, threshold)

if __name__ == "__main__":
    main()