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
import time
import logging
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variable to store the face analysis model
face_app = None
model_lock = Lock()
processing_stats = {
    'total_processed': 0,
    'successful_extractions': 0,
    'failed_extractions': 0,
    'average_processing_time': 0
}

def download_model_manually():
    """Manually download and setup InsightFace model"""
    model_dir = Path.home() / '.insightface' / 'models'
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    model_path = model_dir / "buffalo_l.zip"
    
    try:
        logger.info("Downloading InsightFace model...")
        response = requests.get(model_url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Download with progress indication
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        logger.info(f"Download progress: {progress:.1f}%")
        
        logger.info("Extracting model files...")
        with zipfile.ZipFile(model_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        
        # Clean up zip file
        model_path.unlink()
        logger.info("Model downloaded and extracted successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        return False

def initialize_model():
    """Initialize the InsightFace model with thread safety"""
    global face_app
    
    with model_lock:
        if face_app is not None:
            return True
        
        # Check and download model if needed
        model_dir = Path.home() / '.insightface' / 'models' / 'buffalo_l'
        if not model_dir.exists():
            logger.info("Model not found, downloading...")
            if not download_model_manually():
                logger.error("Failed to download model")
                return False
        
        try:
            logger.info("Initializing InsightFace model...")
            face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
            face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("✅ Model initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False

def extract_face_vector_from_image(image):
    """Extract face vector from OpenCV image with enhanced error handling"""
    global face_app, processing_stats
    
    start_time = time.time()
    
    try:
        if face_app is None:
            if not initialize_model():
                raise Exception("Failed to initialize face recognition model")
        
        # Validate image
        if image is None or image.size == 0:
            raise Exception("Invalid or empty image")
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assuming BGR format from OpenCV
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        logger.info(f"Processing image with shape: {image.shape}")
        
        # Detect faces with error handling
        with model_lock:
            faces = face_app.get(image_rgb)
        
        processing_time = time.time() - start_time
        
        if len(faces) == 0:
            logger.warning("No faces detected in the image")
            processing_stats['failed_extractions'] += 1
            processing_stats['total_processed'] += 1
            return None
        
        # Extract vector from the largest face (most confident detection)
        largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        vector = largest_face.normed_embedding
        
        # Update statistics
        processing_stats['successful_extractions'] += 1
        processing_stats['total_processed'] += 1
        processing_stats['average_processing_time'] = (
            (processing_stats['average_processing_time'] * (processing_stats['total_processed'] - 1) + processing_time) 
            / processing_stats['total_processed']
        )
        
        logger.info(f"✅ Face vector extracted successfully in {processing_time:.2f}s, vector length: {len(vector)}")
        return vector.tolist()
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Face extraction error after {processing_time:.2f}s: {e}")
        processing_stats['failed_extractions'] += 1
        processing_stats['total_processed'] += 1
        return None

def validate_image_data(image_data):
    """Validate and preprocess image data"""
    try:
        # Try to decode and validate the image
        image = Image.open(BytesIO(image_data))
        
        # Check image format
        if image.format not in ['JPEG', 'PNG', 'JPG']:
            logger.warning(f"Unsupported image format: {image.format}")
        
        # Check image size
        width, height = image.size
        if width < 50 or height < 50:
            raise Exception(f"Image too small: {width}x{height}. Minimum size is 50x50")
        
        if width > 4000 or height > 4000:
            logger.info(f"Large image detected: {width}x{height}, resizing...")
            # Resize large images to improve processing speed
            max_size = 2000
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Resized to: {new_width}x{new_height}")
        
        return image
        
    except Exception as e:
        logger.error(f"Image validation failed: {e}")
        raise Exception(f"Invalid image data: {str(e)}")

@app.route('/extract-face-vector', methods=['POST'])
def extract_face_vector_api():
    """API endpoint to extract face vector with enhanced error handling"""
    
    request_start_time = time.time()
    
    try:
        # Get the request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        logger.info("📥 Received face vector extraction request")
        
        # Method 1: Image path
        if 'image_path' in data:
            image_path = data['image_path']
            
            if not os.path.exists(image_path):
                return jsonify({
                    'success': False,
                    'error': f'Image not found: {image_path}'
                }), 404
            
            # Read and validate image
            img = cv2.imread(image_path)
            if img is None:
                return jsonify({
                    'success': False,
                    'error': 'Could not read image file - file may be corrupted'
                }), 400
            
            logger.info(f"Processing image from path: {image_path}")
        
        # Method 2: Base64 encoded image
        elif 'image_base64' in data:
            try:
                # Decode base64 image
                image_data = base64.b64decode(data['image_base64'])
                
                # Validate image data
                image = validate_image_data(image_data)
                
                # Convert PIL image to OpenCV format
                img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                logger.info(f"Processing base64 image, size: {image.size}")
                
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
        
        total_processing_time = time.time() - request_start_time
        
        if vector is None:
            return jsonify({
                'success': False,
                'error': 'No face detected in image or face extraction failed',
                'processing_time': round(total_processing_time, 2),
                'suggestions': [
                    'Ensure the image contains a clear, visible face',
                    'Check that the face is well-lit and not blurry',
                    'Make sure the face is not too small in the image',
                    'Try a different angle or lighting condition'
                ]
            }), 400
        
        # Return success response
        return jsonify({
            'success': True,
            'vector': vector,
            'vector_length': len(vector),
            'message': 'Face vector extracted successfully',
            'processing_time': round(total_processing_time, 2),
            'image_info': {
                'shape': img.shape if img is not None else None,
                'processing_stats': processing_stats
            }
        })
        
    except Exception as e:
        total_processing_time = time.time() - request_start_time
        logger.error(f"API error after {total_processing_time:.2f}s: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}',
            'processing_time': round(total_processing_time, 2)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': face_app is not None,
        'model_info': {
            'providers': ['CPUExecutionProvider'],
            'det_size': (640, 640)
        },
        'processing_stats': processing_stats,
        'uptime': time.time(),
        'version': '2.0'
    })

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get processing statistics"""
    return jsonify({
        'processing_stats': processing_stats,
        'success_rate': (
            processing_stats['successful_extractions'] / processing_stats['total_processed'] * 100
            if processing_stats['total_processed'] > 0 else 0
        ),
        'timestamp': time.time()
    })

@app.route('/reset-stats', methods=['POST'])
def reset_stats():
    """Reset processing statistics"""
    global processing_stats
    processing_stats = {
        'total_processed': 0,
        'successful_extractions': 0,
        'failed_extractions': 0,
        'average_processing_time': 0
    }
    return jsonify({
        'message': 'Statistics reset successfully',
        'processing_stats': processing_stats
    })

@app.route('/', methods=['GET'])
def home():
    """Home endpoint with usage instructions"""
    return jsonify({
        'message': 'Enhanced Face Vector Extraction API v2.0',
        'features': [
            'Thread-safe model loading',
            'Enhanced error handling and validation',
            'Processing statistics tracking',
            'Image preprocessing and optimization',
            'Detailed logging and monitoring'
        ],
        'endpoints': {
            '/extract-face-vector': 'POST - Extract face vector from image',
            '/health': 'GET - Health check with detailed info',
            '/stats': 'GET - Get processing statistics',
            '/reset-stats': 'POST - Reset processing statistics'
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
        },
        'processing_stats': processing_stats
    })

if __name__ == "__main__":
    print("🚀 Starting Enhanced Face Vector Extraction API v2.0...")
    print("📊 Features: Thread-safe, Statistics tracking, Enhanced validation")
    
    # Initialize model on startup
    print("🔄 Initializing model...")
    if initialize_model():
        print("✅ Model ready!")
        print("🎯 API ready for biometric processing")
    else:
        print("❌ Model initialization failed!")
        print("⚠️  API will attempt to initialize on first request")
    
    print(f"🌐 Starting server on http://0.0.0.0:5000")
    print("📝 Check /health for status, /stats for processing statistics")
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)