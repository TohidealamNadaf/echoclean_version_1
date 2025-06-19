from app import db
from datetime import datetime

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    reference_filename = db.Column(db.String(255), nullable=False)
    target_filename = db.Column(db.String(255), nullable=False)
    similarity_score = db.Column(db.Float, nullable=False)
    result = db.Column(db.String(100), nullable=False)
    confidence = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Analysis {self.id}: {self.result} ({self.similarity_score:.4f})>'

class AudioFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_type = db.Column(db.String(10), nullable=False)  # 'reference' or 'target'
    file_path = db.Column(db.String(500), nullable=False)
    upload_timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<AudioFile {self.id}: {self.filename} ({self.file_type})>'
