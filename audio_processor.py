import os
import logging
from pydub import AudioSegment
import tempfile
import math
import array

class AudioProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("AudioProcessor initialized with basic audio feature extraction")
    
    def _preprocess_audio(self, audio_path):
        """Preprocess audio file to the required format"""
        try:
            # Load audio with pydub
            audio = AudioSegment.from_file(audio_path)
            
            # Convert to mono and normalize sample rate
            audio = audio.set_channels(1).set_frame_rate(16000)
            
            # Remove silence
            audio = self._remove_silence(audio)
            
            # Ensure minimum length (1 second)
            if len(audio) < 1000:  # 1000ms = 1s
                silence = AudioSegment.silent(duration=1000 - len(audio))
                audio = audio + silence
            
            # Get raw audio data
            raw_data = audio.raw_data
            sample_width = audio.sample_width
            frame_rate = audio.frame_rate
            
            return raw_data, sample_width, frame_rate
            
        except Exception as e:
            self.logger.error(f"Error preprocessing audio {audio_path}: {str(e)}")
            return None, None, None
    
    def _remove_silence(self, audio_segment, silence_thresh=-50.0, chunk_size=10):
        """Remove silence from audio"""
        try:
            # Split audio into chunks and filter out silent ones
            chunks = [audio_segment[i:i+chunk_size] for i in range(0, len(audio_segment), chunk_size)]
            non_silent_chunks = [chunk for chunk in chunks if chunk.dBFS > silence_thresh]
            
            if not non_silent_chunks:
                return audio_segment  # Return original if all chunks are silent
            
            # Concatenate non-silent chunks
            result = non_silent_chunks[0]
            for chunk in non_silent_chunks[1:]:
                result += chunk
            
            return result
        except:
            return audio_segment
    
    def extract_embedding(self, audio_path):
        """Extract speaker embedding from audio file using basic audio features"""
        try:
            raw_data, sample_width, frame_rate = self._preprocess_audio(audio_path)
            if raw_data is None:
                return None
            
            return self._extract_basic_features(raw_data, sample_width, frame_rate)
                
        except Exception as e:
            self.logger.error(f"Error extracting embedding from {audio_path}: {str(e)}")
            return None
    
    def _extract_basic_features(self, raw_data, sample_width, frame_rate):
        """Extract basic audio features for speaker verification"""
        try:
            # Convert raw data to samples
            if sample_width == 1:
                samples = array.array('b', raw_data)
            elif sample_width == 2:
                samples = array.array('h', raw_data)
            else:
                samples = array.array('i', raw_data)
            
            # Convert to float and normalize
            max_val = float(2 ** (sample_width * 8 - 1))
            normalized_samples = [float(s) / max_val for s in samples]
            
            # Extract basic statistical features
            features = []
            
            # Time domain features
            mean_val = sum(normalized_samples) / len(normalized_samples)
            variance = sum((x - mean_val) ** 2 for x in normalized_samples) / len(normalized_samples)
            std_dev = math.sqrt(variance)
            
            # Zero crossing rate
            zero_crossings = sum(1 for i in range(len(normalized_samples) - 1) 
                               if normalized_samples[i] * normalized_samples[i + 1] < 0)
            zcr = zero_crossings / len(normalized_samples)
            
            # Energy and power
            energy = sum(x ** 2 for x in normalized_samples)
            rms = math.sqrt(energy / len(normalized_samples))
            
            # Spectral features (basic frequency analysis)
            frame_size = 1024
            hop_size = 512
            spectral_features = []
            
            for i in range(0, len(normalized_samples) - frame_size, hop_size):
                frame = normalized_samples[i:i + frame_size]
                
                # Simple spectral centroid approximation
                magnitudes = [abs(x) for x in frame]
                total_magnitude = sum(magnitudes)
                
                if total_magnitude > 0:
                    spectral_centroid = sum(i * mag for i, mag in enumerate(magnitudes)) / total_magnitude
                    spectral_features.append(spectral_centroid)
            
            # Statistical measures of spectral features
            if spectral_features:
                spec_mean = sum(spectral_features) / len(spectral_features)
                spec_var = sum((x - spec_mean) ** 2 for x in spectral_features) / len(spectral_features)
                spec_std = math.sqrt(spec_var)
            else:
                spec_mean = spec_std = 0
            
            # Compile feature vector
            features = [
                mean_val, std_dev, zcr, rms, energy,
                spec_mean, spec_std,
                max(normalized_samples), min(normalized_samples),
                len(normalized_samples) / frame_rate  # Duration
            ]
            
            # Normalize feature vector
            feature_magnitude = math.sqrt(sum(f ** 2 for f in features))
            if feature_magnitude > 0:
                features = [f / feature_magnitude for f in features]
            
            self.logger.debug(f"Basic features extracted: {len(features)} dimensions")
            return features
            
        except Exception as e:
            self.logger.error(f"Error with basic feature extraction: {str(e)}")
            return None
    
    def calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        try:
            if not embedding1 or not embedding2 or len(embedding1) != len(embedding2):
                return 0.0
            
            # Calculate dot product
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            
            # Calculate magnitudes
            magnitude1 = math.sqrt(sum(a ** 2 for a in embedding1))
            magnitude2 = math.sqrt(sum(b ** 2 for b in embedding2))
            
            # Calculate cosine similarity
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            similarity = dot_product / (magnitude1 * magnitude2)
            
            # Ensure similarity is between 0 and 1
            similarity = max(0.0, min(1.0, similarity))
            
            self.logger.debug(f"Calculated similarity: {similarity:.4f}")
            return similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def validate_audio_file(self, audio_path):
        """Validate that the audio file is readable and has content"""
        try:
            result = self._preprocess_audio(audio_path)
            if result[0] is None:
                return False, "Audio file is empty or corrupted"
            
            raw_data, sample_width, frame_rate = result
            if len(raw_data) == 0:
                return False, "Audio file is empty or corrupted"
            
            duration_ms = len(raw_data) / (sample_width * frame_rate) * 1000
            if duration_ms < 500:  # Less than 0.5 seconds
                return False, "Audio file is too short (minimum 0.5 seconds required)"
            
            return True, "Audio file is valid"
            
        except Exception as e:
            return False, f"Audio validation failed: {str(e)}"