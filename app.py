import os
import logging
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
from audio_processor import AudioProcessor
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "echoclean-secret-key-2025")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac', 'm4a', 'ogg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Configure database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///echoclean.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Initialize extensions
db.init_app(app)

# Initialize audio processor
audio_processor = AudioProcessor()

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-reference', methods=['POST'])
def upload_reference():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Supported: WAV, MP3, FLAC, M4A, OGG'}), 400
        
        # Generate unique filename
        filename = secure_filename(f"ref_{uuid.uuid4().hex}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract embedding
        embedding = audio_processor.extract_embedding(filepath)
        if embedding is None:
            os.remove(filepath)
            return jsonify({'error': 'Failed to process audio file. Please check the file format and quality.'}), 400
        
        # Store reference embedding in session or database
        # For simplicity, we'll store in session
        from flask import session
        session['reference_file'] = filepath
        session['reference_embedding'] = embedding if isinstance(embedding, list) else embedding.tolist()
        
        app.logger.info(f"Reference file uploaded: {filename}")
        return jsonify({
            'success': True,
            'message': 'Reference voice uploaded successfully',
            'filename': filename
        })
        
    except Exception as e:
        app.logger.error(f"Error uploading reference: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/upload-target', methods=['POST'])
def upload_target():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Supported: WAV, MP3, FLAC, M4A, OGG'}), 400
        
        # Generate unique filename
        filename = secure_filename(f"target_{uuid.uuid4().hex}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract embedding
        embedding = audio_processor.extract_embedding(filepath)
        if embedding is None:
            os.remove(filepath)
            return jsonify({'error': 'Failed to process audio file. Please check the file format and quality.'}), 400
        
        app.logger.info(f"Target file uploaded: {filename}")
        return jsonify({
            'success': True,
            'message': 'Target voice uploaded successfully',
            'filename': filename,
            'embedding': embedding if isinstance(embedding, list) else embedding.tolist()
        })
        
    except Exception as e:
        app.logger.error(f"Error uploading target: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        from flask import session
        
        # Check if reference embedding exists
        if 'reference_embedding' not in session:
            return jsonify({'error': 'No reference voice uploaded. Please upload a reference file first.'}), 400
        
        data = request.get_json()
        if not data or 'target_embedding' not in data:
            return jsonify({'error': 'No target embedding provided'}), 400
        
        reference_embedding = session['reference_embedding']
        target_embedding = data['target_embedding']
        
        # Calculate similarity
        similarity = audio_processor.calculate_similarity(reference_embedding, target_embedding)
        
        # Apply enhanced threshold logic for better deepfake detection
        if similarity >= 0.85:
            result = "✅ Authentic"
            confidence = "High"
            color = "success"
        elif 0.65 <= similarity < 0.85:
            result = "⚠️ Possibly Different Speaker"
            confidence = "Medium"
            color = "warning"
        elif 0.35 <= similarity < 0.65:
            result = "❌ Different Speaker"
            confidence = "High"
            color = "danger"
        else:
            result = "🚨 Deepfake or Synthetic Audio"
            confidence = "Very High"
            color = "dark"
        
        # Log analysis for potential CSV export
        app.logger.info(f"Analysis result: {result}, Similarity: {similarity:.4f}")
        
        return jsonify({
            'success': True,
            'similarity': round(similarity, 4),
            'result': result,
            'confidence': confidence,
            'color': color,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error during analysis: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/compare-realtime', methods=['POST'])
def compare_realtime():
    """Real-time voice comparison for instant feedback"""
    try:
        from flask import session
        
        # Check if reference embedding exists
        if 'reference_embedding' not in session:
            return jsonify({'error': 'No reference voice uploaded. Please upload a reference file first.'}), 400
        
        if 'file' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Generate temporary filename for real-time comparison
        filename = secure_filename(f"realtime_{uuid.uuid4().hex}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract embedding from uploaded audio
            target_embedding = audio_processor.extract_embedding(filepath)
            if target_embedding is None:
                return jsonify({'error': 'Failed to process audio file'}), 400
            
            # Get reference embedding
            reference_embedding = session['reference_embedding']
            
            # Calculate similarity
            similarity = audio_processor.calculate_similarity(reference_embedding, target_embedding)
            
            # Apply enhanced threshold logic for real-time deepfake detection
            if similarity >= 0.85:
                result = "✅ Authentic"
                confidence = "High"
                color = "success"
            elif 0.65 <= similarity < 0.85:
                result = "⚠️ Possibly Different Speaker"
                confidence = "Medium"
                color = "warning"
            elif 0.35 <= similarity < 0.65:
                result = "❌ Different Speaker"
                confidence = "High"
                color = "danger"
            else:
                result = "🚨 Deepfake or Synthetic Audio"
                confidence = "Very High"
                color = "dark"
            
            app.logger.info(f"Real-time comparison: {result}, Similarity: {similarity:.4f}")
            
            return jsonify({
                'success': True,
                'similarity': round(similarity, 4),
                'result': result,
                'confidence': confidence,
                'color': color,
                'realtime': True
            })
            
        finally:
            # Clean up temporary file
            try:
                os.remove(filepath)
            except:
                pass
        
    except Exception as e:
        app.logger.error(f"Error during real-time comparison: {str(e)}")
        return jsonify({'error': f'Real-time comparison failed: {str(e)}'}), 500

@app.route('/clear-session', methods=['POST'])
def clear_session():
    try:
        from flask import session
        
        # Clean up uploaded files
        if 'reference_file' in session:
            try:
                os.remove(session['reference_file'])
            except:
                pass
        
        # Clear session
        session.clear()
        
        return jsonify({'success': True, 'message': 'Session cleared successfully'})
        
    except Exception as e:
        app.logger.error(f"Error clearing session: {str(e)}")
        return jsonify({'error': f'Failed to clear session: {str(e)}'}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(e):
    app.logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error. Please try again.'}), 500

# Initialize database
with app.app_context():
    import models
    db.create_all()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
