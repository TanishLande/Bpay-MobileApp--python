import os
import requests
from pathlib import Path
import zipfile
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Global variable to store the face analysis model
face_app = None

def download_model_manually():
    """Manually download and setup InsightFace model"""
    model_dir = Path.home() / '.insightface' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    model_path = model_dir / "buffalo_l.zip"
    
    try:
        print("Downloading model...")
        response = requests.get(model_url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        
        model_path.unlink()  # Remove zip file
        print("Model downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Model download failed: {e}")
        return False

def initialize_model():
    """Initialize the InsightFace model"""
    global face_app
    
    if face_app is not None:
        return True
    
    # Check and download model if needed
    model_dir = Path.home() / '.insightface' / 'models' / 'buffalo_l'
    if not model_dir.exists():
        if not download_model_manually():
            return False
    
    try:
        face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_size=(640, 640))
        print("Model initialized successfully!")
        return True
    except Exception as e:
        print(f"Model initialization failed: {e}")
        return False

def extract_face_vector_from_image(image):
    """Extract face vector from OpenCV image"""
    global face_app
    
    if face_app is None:
        if not initialize_model():
            return None
    
    try:
        # Detect faces
        faces = face_app.get(image)
        
        if len(faces) == 0:
            return None
        
        # Extract vector from first face
        vector = faces[0].normed_embedding
        return vector.tolist()  # Convert numpy array to list for JSON serialization
        
    except Exception as e:
        print(f"Face extraction error: {e}")
        return None

@app.route('/extract-face-vector', methods=['POST'])
def extract_face_vector_api():
    """API endpoint to extract face vector"""
    
    try:
        # Get the request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Method 1: Image path
        if 'image_path' in data:
            image_path = data['image_path']
            
            if not os.path.exists(image_path):
                return jsonify({
                    'success': False,
                    'error': f'Image not found: {image_path}'
                }), 404
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return jsonify({
                    'success': False,
                    'error': 'Could not read image file'
                }), 400
        
        # Method 2: Base64 encoded image
        elif 'image_base64' in data:
            try:
                # Decode base64 image
                image_data = base64.b64decode(data['image_base64'])
                image = Image.open(BytesIO(image_data))
                # Convert PIL image to OpenCV format
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Invalid base64 image: {str(e)}'
                }), 400
        
        else:
            return jsonify({
                'success': False,
                'error': 'Please provide either image_path or image_base64'
            }), 400
        
        # Extract face vector
        vector = extract_face_vector_from_image(img)
        
        if vector is None:
            return jsonify({
                'success': False,
                'error': 'No face detected in image'
            }), 400
        
        # Return success response
        return jsonify({
            'success': True,
            'vector': vector,
            'vector_length': len(vector),
            'message': 'Face vector extracted successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': face_app is not None
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with usage instructions"""
    return jsonify({
        'message': 'Face Vector Extraction API',
        'endpoints': {
            '/extract-face-vector': 'POST - Extract face vector from image',
            '/health': 'GET - Health check'
        },
        'usage': {
            'method1': {
                'description': 'Send image file path',
                'example': {
                    'image_path': '/path/to/image.jpg'
                }
            },
            'method2': {
                'description': 'Send base64 encoded image',
                'example': {
                    'image_base64': 'iVBORw0KGgoAAAANSUhEUgAA...'
                }
            }
        }
    })

if __name__ == "__main__":
    print("Starting Face Vector Extraction API...")
    
    # Initialize model on startup
    print("Initializing model...")
    if initialize_model():
        print("✓ Model ready!")
    else:
        print("✗ Model initialization failed!")
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, debug=True)