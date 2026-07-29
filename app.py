"""
ClosetCraft: AI Fashion Recommendation & Outfit Generator
Flask Web Application
"""

from flask import Flask, render_template, request, jsonify, url_for
import os
import pickle
from werkzeug.utils import secure_filename
from model.feature_extraction import FashionFeatureExtractor, extract_dominant_color, colors_match

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATASET_FOLDER'] = 'static/images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize feature extractor
feature_extractor = FashionFeatureExtractor()

# Load or build dataset features
dataset_features = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def initialize_dataset():
    """Initialize dataset features on startup"""
    global dataset_features
    dataset_features = feature_extractor.load_dataset_features()
    
    if dataset_features is None:
        print("Building dataset features...")
        dataset_features = feature_extractor.build_dataset_features(
            dataset_path=app.config['DATASET_FOLDER'],
            save_path='dataset_features.pkl'
        )
    else:
        print(f"Loaded {len(dataset_features)} pre-computed features")
    
    return dataset_features

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    """Handle fashion recommendation request"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Find similar items
        similar_items = feature_extractor.find_similar_items(
            filepath, 
            dataset_features, 
            top_k=5
        )
        
        # Prepare results
        recommendations = []
        for item_name, similarity in similar_items:
            recommendations.append({
                'filename': item_name,
                'similarity': round(float(similarity) * 100, 2),
                'image_url': url_for('static', filename=f'images/{item_name}')
            })
        
        return jsonify({
            'success': True,
            'uploaded_image': url_for('static', filename=f'uploads/{filename}'),
            'recommendations': recommendations
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_outfit', methods=['POST'])
def generate_outfit():
    """Handle outfit generation request"""
    uploaded_items = {}
    
    # Check for multiple file uploads
    for category in ['top', 'bottom', 'shoes', 'accessory']:
        if category in request.files:
            file = request.files[category]
            if file and file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(f"{category}_{file.filename}")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_items[category] = filepath
    
    if len(uploaded_items) < 2:
        return jsonify({'error': 'Please upload at least 2 clothing items (e.g., top and bottom)'}), 400
    
    try:
        # Generate outfit suggestion
        outfit = feature_extractor.generate_outfit(uploaded_items)
        
        # Add color matching analysis (bonus feature)
        color_analysis = []
        colors = {}
        
        for category, filepath in uploaded_items.items():
            dominant_color = extract_dominant_color(filepath)
            colors[category] = dominant_color
        
        # Check color compatibility
        color_comments = []
        if 'top' in colors and 'bottom' in colors:
            if colors_match(colors['top'], colors['bottom']):
                color_comments.append("✓ Top and bottom colors complement each other")
            else:
                color_comments.append("⚠️ Consider adjusting top or bottom for better color harmony")
        
        if 'shoes' in colors and 'bottom' in colors:
            if colors_match(colors['shoes'], colors['bottom']):
                color_comments.append("✓ Shoes match well with the bottom")
        
        # Prepare response
        response = {
            'success': True,
            'outfit': outfit,
            'color_analysis': color_comments,
            'items': {cat: url_for('static', filename=f'uploads/{os.path.basename(path)}') 
                     for cat, path in uploaded_items.items()}
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/build_features', methods=['POST'])
def build_features():
    """Rebuild dataset features (admin functionality)"""
    global dataset_features
    try:
        dataset_features = feature_extractor.build_dataset_features(
            dataset_path=app.config['DATASET_FOLDER'],
            save_path='dataset_features.pkl'
        )
        return jsonify({
            'success': True,
            'message': f'Features rebuilt for {len(dataset_features)} images'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['DATASET_FOLDER'], exist_ok=True)
    
    # Initialize dataset
    initialize_dataset()
    
    # Run Flask app
    app.run(debug=True, host='127.0.0.1', port=5000)