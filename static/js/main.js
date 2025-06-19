// EchoClean Frontend JavaScript

class EchoClean {
    constructor() {
        this.referenceUploaded = false;
        this.targetEmbedding = null;
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.recordedBlob = null;
        this.initializeEventListeners();
        this.initializeToast();
    }

    initializeEventListeners() {
        // File input listeners
        document.getElementById('referenceFile').addEventListener('change', () => {
            this.handleFileSelection('reference');
        });

        document.getElementById('targetFile').addEventListener('change', () => {
            this.handleFileSelection('target');
        });

        // Button listeners
        document.getElementById('uploadReferenceBtn').addEventListener('click', () => {
            this.uploadReference();
        });

        document.getElementById('uploadTargetBtn').addEventListener('click', () => {
            this.uploadTarget();
        });

        document.getElementById('analyzeBtn').addEventListener('click', () => {
            this.analyzeVoices();
        });

        document.getElementById('clearBtn').addEventListener('click', () => {
            this.clearSession();
        });

        // Recording button listeners
        document.getElementById('recordBtn').addEventListener('click', () => {
            this.startRecording();
        });

        document.getElementById('stopRecordBtn').addEventListener('click', () => {
            this.stopRecording();
        });

        document.getElementById('realtimeCompareBtn').addEventListener('click', () => {
            this.realtimeCompare();
        });
    }

    initializeToast() {
        this.toast = new bootstrap.Toast(document.getElementById('toast'));
    }

    handleFileSelection(type) {
        const fileInput = document.getElementById(`${type}File`);
        const uploadBtn = document.getElementById(`upload${type.charAt(0).toUpperCase() + type.slice(1)}Btn`);
        
        if (fileInput.files.length > 0) {
            const file = fileInput.files[0];
            if (this.validateFile(file)) {
                uploadBtn.disabled = false;
                this.updateStatus(type, `Selected: ${file.name}`, 'info');
            } else {
                uploadBtn.disabled = true;
                fileInput.value = '';
                this.updateStatus(type, 'Invalid file format or size', 'error');
            }
        } else {
            uploadBtn.disabled = true;
            this.updateStatus(type, '', '');
        }
    }

    validateFile(file) {
        const allowedTypes = ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/mp4', 'audio/ogg'];
        const maxSize = 16 * 1024 * 1024; // 16MB

        if (!allowedTypes.includes(file.type) && !file.name.match(/\.(wav|mp3|flac|m4a|ogg)$/i)) {
            this.showToast('Invalid file format. Please use WAV, MP3, FLAC, M4A, or OGG files.', 'error');
            return false;
        }

        if (file.size > maxSize) {
            this.showToast('File too large. Maximum size is 16MB.', 'error');
            return false;
        }

        return true;
    }

    async uploadReference() {
        const fileInput = document.getElementById('referenceFile');
        const uploadBtn = document.getElementById('uploadReferenceBtn');
        
        if (!fileInput.files.length) {
            this.showToast('Please select a reference file first.', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Uploading...';
        this.updateStatus('reference', 'Uploading and processing...', 'info');

        try {
            const response = await fetch('/upload-reference', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.referenceUploaded = true;
                this.updateStatus('reference', '✅ Reference voice uploaded successfully', 'success');
                this.showToast('Reference voice uploaded and processed successfully!', 'success');
                this.updateAnalyzeButton();
            } else {
                this.updateStatus('reference', `❌ ${data.error}`, 'error');
                this.showToast(data.error, 'error');
            }
        } catch (error) {
            this.updateStatus('reference', '❌ Upload failed', 'error');
            this.showToast('Upload failed. Please try again.', 'error');
            console.error('Upload error:', error);
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = '<i class="fas fa-upload me-2"></i>Upload Reference';
        }
    }



    async analyzeVoices() {
        if (!this.referenceUploaded || !this.targetEmbedding) {
            this.showToast('Please upload both reference and target voice files first.', 'error');
            return;
        }

        const analyzeBtn = document.getElementById('analyzeBtn');
        const progressDiv = document.getElementById('analysisProgress');
        
        analyzeBtn.disabled = true;
        progressDiv.style.display = 'block';

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    target_embedding: this.targetEmbedding
                })
            });

            const data = await response.json();

            if (data.success) {
                this.displayResults(data);
                this.showToast('Analysis completed successfully!', 'success');
            } else {
                this.showToast(data.error, 'error');
            }
        } catch (error) {
            this.showToast('Analysis failed. Please try again.', 'error');
            console.error('Analysis error:', error);
        } finally {
            analyzeBtn.disabled = false;
            progressDiv.style.display = 'none';
        }
    }

    displayResults(data) {
        const resultsSection = document.getElementById('resultsSection');
        const similarityScore = document.getElementById('similarityScore');
        const similarityBar = document.getElementById('similarityBar');
        const resultText = document.getElementById('resultText');
        const resultAlert = document.getElementById('resultAlert');
        const resultDescription = document.getElementById('resultDescription');

        // Update similarity score
        similarityScore.textContent = data.similarity.toFixed(3);
        
        // Update progress bar
        const percentage = (data.similarity * 100).toFixed(1);
        similarityBar.style.width = `${percentage}%`;
        similarityBar.setAttribute('aria-valuenow', percentage);
        
        // Set progress bar color based on result
        similarityBar.className = `progress-bar similarity-${data.color === 'success' ? 'authentic' : 
                                                                data.color === 'warning' ? 'possible' : 
                                                                data.color === 'danger' ? 'different' : 'deepfake'}`;

        // Update result text
        resultText.textContent = data.result;
        resultText.className = `text-${data.color}`;

        // Update result alert
        resultAlert.className = `alert alert-${data.color}`;
        resultAlert.style.display = 'block';
        
        // Set description based on result
        let description = '';
        if (data.similarity >= 0.90) {
            description = 'High confidence match. The voices are likely from the same person.';
        } else if (data.similarity >= 0.75) {
            description = 'Moderate confidence. The voices may be from different people or contain variations.';
        } else if (data.similarity >= 0.50) {
            description = 'Low similarity. The voices are likely from different speakers.';
        } else {
            description = 'Very low similarity. This may indicate a deepfake, voice synthesis, or corrupted audio.';
        }
        
        resultDescription.textContent = description;

        // Show results section with animation
        resultsSection.style.display = 'block';
        resultsSection.classList.add('fade-in');

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    async clearSession() {
        try {
            const response = await fetch('/clear-session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });

            const data = await response.json();

            if (data.success) {
                this.resetInterface();
                this.showToast('Session cleared successfully!', 'success');
            } else {
                this.showToast(data.error, 'error');
            }
        } catch (error) {
            this.showToast('Failed to clear session. Please refresh the page.', 'error');
            console.error('Clear session error:', error);
        }
    }

    resetInterface() {
        // Reset state
        this.referenceUploaded = false;
        this.targetEmbedding = null;

        // Reset file inputs
        document.getElementById('referenceFile').value = '';
        document.getElementById('targetFile').value = '';

        // Reset buttons
        document.getElementById('uploadReferenceBtn').disabled = true;
        document.getElementById('uploadTargetBtn').disabled = true;
        document.getElementById('analyzeBtn').disabled = true;

        // Clear status messages
        this.updateStatus('reference', '', '');
        this.updateStatus('target', '', '');

        // Hide results
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('analysisProgress').style.display = 'none';

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    updateStatus(type, message, status) {
        const statusDiv = document.getElementById(`${type}Status`);
        statusDiv.innerHTML = message;
        statusDiv.className = status ? `status-${status}` : '';
    }

    updateAnalyzeButton() {
        const analyzeBtn = document.getElementById('analyzeBtn');
        analyzeBtn.disabled = !(this.referenceUploaded && this.targetEmbedding);
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    volume: 1.0
                } 
            });
            
            this.recordedChunks = [];
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.recordedChunks.push(event.data);
                }
            };
            
            this.mediaRecorder.onstop = () => {
                this.recordedBlob = new Blob(this.recordedChunks, { type: 'audio/webm' });
                this.handleRecordingComplete();
                
                // Stop all tracks to release microphone
                stream.getTracks().forEach(track => track.stop());
            };
            
            this.mediaRecorder.start();
            
            // Update UI
            document.getElementById('recordBtn').disabled = true;
            document.getElementById('stopRecordBtn').disabled = false;
            this.updateStatus('recording', 'Recording... Speak now!', 'info');
            
            this.showToast('Recording started. Speak clearly into your microphone.', 'info');
            
        } catch (error) {
            this.showToast('Failed to access microphone. Please allow microphone access.', 'error');
            console.error('Recording error:', error);
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.mediaRecorder.state !== 'inactive') {
            this.mediaRecorder.stop();
            
            // Update UI
            document.getElementById('recordBtn').disabled = false;
            document.getElementById('stopRecordBtn').disabled = true;
            this.updateStatus('recording', 'Processing recording...', 'info');
        }
    }

    handleRecordingComplete() {
        if (!this.recordedBlob) return;
        
        // Create audio playback
        const audioPlayback = document.getElementById('audioPlayback');
        const audioUrl = URL.createObjectURL(this.recordedBlob);
        audioPlayback.src = audioUrl;
        audioPlayback.style.display = 'block';
        
        // Enable upload button for recorded audio
        document.getElementById('uploadTargetBtn').disabled = false;
        
        // Enable real-time compare if reference is uploaded
        if (this.referenceUploaded) {
            document.getElementById('realtimeCompareBtn').disabled = false;
        }
        
        this.updateStatus('recording', '✅ Recording complete! You can upload or compare instantly.', 'success');
        this.showToast('Recording completed successfully!', 'success');
    }

    async uploadTarget() {
        const fileInput = document.getElementById('targetFile');
        const uploadBtn = document.getElementById('uploadTargetBtn');
        
        let formData = new FormData();
        
        // Check if we have a recorded audio or file upload
        if (this.recordedBlob) {
            // Convert recorded blob to WAV and upload
            formData.append('file', this.recordedBlob, 'recorded_audio.webm');
        } else if (fileInput.files.length > 0) {
            formData.append('file', fileInput.files[0]);
        } else {
            this.showToast('Please select a file or record audio first.', 'error');
            return;
        }

        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Uploading...';
        this.updateStatus('target', 'Uploading and processing...', 'info');

        try {
            const response = await fetch('/upload-target', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                this.targetEmbedding = data.embedding;
                this.updateStatus('target', '✅ Target voice uploaded successfully', 'success');
                this.showToast('Target voice uploaded and processed successfully!', 'success');
                this.updateAnalyzeButton();
            } else {
                this.updateStatus('target', `❌ ${data.error}`, 'error');
                this.showToast(data.error, 'error');
            }
        } catch (error) {
            this.updateStatus('target', '❌ Upload failed', 'error');
            this.showToast('Upload failed. Please try again.', 'error');
            console.error('Upload error:', error);
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = '<i class="fas fa-upload me-2"></i>Upload Test Voice';
        }
    }

    updateStatus(type, message, status) {
        const statusDiv = document.getElementById(`${type}Status`);
        if (statusDiv) {
            statusDiv.innerHTML = message;
            statusDiv.className = status ? `status-${status}` : '';
        }
        
        // Also update recording status if it's a recording update
        if (type === 'recording') {
            const recordingStatusDiv = document.getElementById('recordingStatus');
            if (recordingStatusDiv) {
                recordingStatusDiv.innerHTML = message;
                recordingStatusDiv.className = status ? `status-${status}` : '';
            }
        }
    }

    resetInterface() {
        // Reset state
        this.referenceUploaded = false;
        this.targetEmbedding = null;
        this.recordedBlob = null;
        this.recordedChunks = [];

        // Reset file inputs
        document.getElementById('referenceFile').value = '';
        document.getElementById('targetFile').value = '';

        // Reset buttons
        document.getElementById('uploadReferenceBtn').disabled = true;
        document.getElementById('uploadTargetBtn').disabled = true;
        document.getElementById('analyzeBtn').disabled = true;
        document.getElementById('recordBtn').disabled = false;
        document.getElementById('stopRecordBtn').disabled = true;
        document.getElementById('realtimeCompareBtn').disabled = true;

        // Clear status messages
        this.updateStatus('reference', '', '');
        this.updateStatus('target', '', '');
        this.updateStatus('recording', '', '');

        // Hide audio playback
        document.getElementById('audioPlayback').style.display = 'none';

        // Hide results
        document.getElementById('resultsSection').style.display = 'none';
        document.getElementById('analysisProgress').style.display = 'none';

        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    async realtimeCompare() {
        if (!this.referenceUploaded) {
            this.showToast('Please upload a reference voice first.', 'error');
            return;
        }

        if (!this.recordedBlob) {
            this.showToast('Please record audio first.', 'error');
            return;
        }

        const compareBtn = document.getElementById('realtimeCompareBtn');
        compareBtn.disabled = true;
        compareBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Comparing...';

        try {
            const formData = new FormData();
            formData.append('file', this.recordedBlob, 'realtime_audio.webm');

            const response = await fetch('/compare-realtime', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                // Show instant results in a toast and brief display
                this.showToast(`Real-time result: ${data.result} (${(data.similarity * 100).toFixed(1)}%)`, 
                              data.color === 'success' ? 'success' : 'warning');
                
                // Update recording status with instant feedback
                this.updateStatus('recording', 
                    `🚀 Real-time: ${data.result} - ${(data.similarity * 100).toFixed(1)}% match`, 
                    data.color === 'success' ? 'success' : 
                    data.color === 'warning' ? 'warning' : 'error');
                
                // Optionally show full results if available
                if (document.getElementById('resultsSection').style.display === 'none') {
                    this.displayResults(data);
                }
            } else {
                this.showToast(data.error, 'error');
            }
        } catch (error) {
            this.showToast('Real-time comparison failed. Please try again.', 'error');
            console.error('Real-time comparison error:', error);
        } finally {
            compareBtn.disabled = false;
            compareBtn.innerHTML = '<i class="fas fa-bolt me-2"></i>Real-time Compare';
        }
    }

    showToast(message, type = 'info') {
        const toastBody = document.getElementById('toastBody');
        const toastIcon = document.querySelector('#toast .toast-header i');
        
        toastBody.textContent = message;
        
        // Update icon based on type
        const iconClass = type === 'success' ? 'fa-check-circle text-success' :
                         type === 'error' ? 'fa-exclamation-circle text-danger' :
                         type === 'warning' ? 'fa-exclamation-triangle text-warning' :
                         'fa-info-circle text-primary';
        
        toastIcon.className = `fas ${iconClass} me-2`;
        
        this.toast.show();
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new EchoClean();
});
