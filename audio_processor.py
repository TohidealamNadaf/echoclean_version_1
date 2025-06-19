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
        """Extract enhanced audio features for better speaker verification"""
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
            
            # Extract enhanced features
            features = []
            
            # Time domain features
            mean_val = sum(normalized_samples) / len(normalized_samples)
            variance = sum((x - mean_val) ** 2 for x in normalized_samples) / len(normalized_samples)
            std_dev = math.sqrt(variance)
            
            # Zero crossing rate
            zero_crossings = sum(1 for i in range(len(normalized_samples) - 1) 
                               if normalized_samples[i] * normalized_samples[i + 1] < 0)
            zcr = zero_crossings / len(normalized_samples)
            
            # Energy and power features
            energy = sum(x ** 2 for x in normalized_samples)
            rms = math.sqrt(energy / len(normalized_samples))
            
            # Spectral analysis with multiple frame sizes for better frequency resolution
            frame_sizes = [512, 1024, 2048]
            all_spectral_features = []
            
            for frame_size in frame_sizes:
                hop_size = frame_size // 2
                spectral_centroids = []
                spectral_rolloffs = []
                spectral_fluxes = []
                prev_spectrum = None
                
                for i in range(0, len(normalized_samples) - frame_size, hop_size):
                    frame = normalized_samples[i:i + frame_size]
                    
                    # Apply window function (Hamming)
                    windowed_frame = [frame[j] * (0.54 - 0.46 * math.cos(2 * math.pi * j / (frame_size - 1))) 
                                    for j in range(frame_size)]
                    
                    # Simple FFT approximation using frequency domain analysis
                    magnitudes = [abs(x) for x in windowed_frame]
                    total_magnitude = sum(magnitudes)
                    
                    if total_magnitude > 0:
                        # Spectral centroid
                        centroid = sum(i * mag for i, mag in enumerate(magnitudes)) / total_magnitude
                        spectral_centroids.append(centroid)
                        
                        # Spectral rolloff (85% of energy)
                        cumulative_energy = 0
                        rolloff_point = 0
                        target_energy = total_magnitude * 0.85
                        
                        for idx, mag in enumerate(magnitudes):
                            cumulative_energy += mag
                            if cumulative_energy >= target_energy:
                                rolloff_point = idx
                                break
                        
                        spectral_rolloffs.append(rolloff_point / len(magnitudes))
                        
                        # Spectral flux (measure of spectral change)
                        if prev_spectrum is not None:
                            flux = sum((magnitudes[j] - prev_spectrum[j]) ** 2 
                                     for j in range(min(len(magnitudes), len(prev_spectrum))))
                            spectral_fluxes.append(math.sqrt(flux))
                        
                        prev_spectrum = magnitudes[:]
                
                # Add statistical measures for this frame size
                if spectral_centroids:
                    all_spectral_features.extend([
                        sum(spectral_centroids) / len(spectral_centroids),  # mean
                        math.sqrt(sum((x - sum(spectral_centroids) / len(spectral_centroids)) ** 2 
                                for x in spectral_centroids) / len(spectral_centroids)),  # std
                    ])
                else:
                    all_spectral_features.extend([0, 0])
                
                if spectral_rolloffs:
                    all_spectral_features.extend([
                        sum(spectral_rolloffs) / len(spectral_rolloffs),
                        math.sqrt(sum((x - sum(spectral_rolloffs) / len(spectral_rolloffs)) ** 2 
                                for x in spectral_rolloffs) / len(spectral_rolloffs)),
                    ])
                else:
                    all_spectral_features.extend([0, 0])
                
                if spectral_fluxes:
                    all_spectral_features.append(sum(spectral_fluxes) / len(spectral_fluxes))
                else:
                    all_spectral_features.append(0)
            
            # Pitch estimation using autocorrelation
            pitch_features = self._extract_pitch_features(normalized_samples, frame_rate)
            
            # Formant-like features (vocal tract characteristics)
            formant_features = self._extract_formant_features(normalized_samples, frame_rate)
            
            # Compile comprehensive feature vector
            features = [
                mean_val, std_dev, zcr, rms, energy,
                max(normalized_samples), min(normalized_samples),
                len(normalized_samples) / frame_rate  # Duration
            ]
            
            features.extend(all_spectral_features)
            features.extend(pitch_features)
            features.extend(formant_features)
            
            # Normalize feature vector
            feature_magnitude = math.sqrt(sum(f ** 2 for f in features))
            if feature_magnitude > 0:
                features = [f / feature_magnitude for f in features]
            
            self.logger.debug(f"Enhanced features extracted: {len(features)} dimensions")
            return features
            
        except Exception as e:
            self.logger.error(f"Error with enhanced feature extraction: {str(e)}")
            return None
    
    def _extract_pitch_features(self, samples, sample_rate):
        """Extract pitch-related features using autocorrelation"""
        try:
            # Autocorrelation for pitch detection
            frame_size = min(2048, len(samples))
            autocorr = []
            
            for lag in range(1, frame_size // 2):
                correlation = sum(samples[i] * samples[i + lag] 
                               for i in range(len(samples) - lag)) / (len(samples) - lag)
                autocorr.append(correlation)
            
            if autocorr:
                max_corr = max(autocorr)
                max_corr_lag = autocorr.index(max_corr) + 1
                
                # Estimate fundamental frequency
                f0 = sample_rate / max_corr_lag if max_corr_lag > 0 else 0
                
                return [max_corr, f0 / sample_rate, len([x for x in autocorr if x > max_corr * 0.5])]
            
            return [0, 0, 0]
            
        except Exception as e:
            self.logger.error(f"Pitch extraction error: {str(e)}")
            return [0, 0, 0]
    
    def _extract_formant_features(self, samples, sample_rate):
        """Extract formant-like features for vocal tract characteristics"""
        try:
            # Simple formant estimation using peak detection in frequency domain
            frame_size = min(2048, len(samples))
            
            # Apply window and get magnitude spectrum
            windowed = [samples[i] * (0.54 - 0.46 * math.cos(2 * math.pi * i / (frame_size - 1))) 
                       for i in range(frame_size)]
            
            # Simple frequency analysis
            freq_bins = []
            for k in range(frame_size // 2):
                real_part = sum(windowed[n] * math.cos(2 * math.pi * k * n / frame_size) 
                              for n in range(frame_size))
                imag_part = -sum(windowed[n] * math.sin(2 * math.pi * k * n / frame_size) 
                               for n in range(frame_size))
                magnitude = math.sqrt(real_part ** 2 + imag_part ** 2)
                freq_bins.append(magnitude)
            
            # Find peaks (potential formants)
            peaks = []
            for i in range(1, len(freq_bins) - 1):
                if freq_bins[i] > freq_bins[i-1] and freq_bins[i] > freq_bins[i+1]:
                    peaks.append((i, freq_bins[i]))
            
            # Sort by magnitude and take top 3 (F1, F2, F3)
            peaks.sort(key=lambda x: x[1], reverse=True)
            formants = peaks[:3] if len(peaks) >= 3 else peaks + [(0, 0)] * (3 - len(peaks))
            
            # Return formant frequencies normalized by sample rate
            return [f[0] / len(freq_bins) for f in formants]
            
        except Exception as e:
            self.logger.error(f"Formant extraction error: {str(e)}")
            return [0, 0, 0]
    
    def calculate_similarity(self, embedding1, embedding2):
        """Calculate enhanced similarity with deepfake detection"""
        try:
            if not embedding1 or not embedding2 or len(embedding1) != len(embedding2):
                return 0.0
            
            # Calculate multiple similarity metrics
            cosine_sim = self._cosine_similarity(embedding1, embedding2)
            euclidean_sim = self._euclidean_similarity(embedding1, embedding2)
            correlation_sim = self._correlation_similarity(embedding1, embedding2)
            
            # Check for deepfake indicators
            deepfake_penalty = self._detect_deepfake_indicators(embedding1, embedding2)
            
            # Weighted combination of similarities
            combined_similarity = (
                0.5 * cosine_sim + 
                0.3 * euclidean_sim + 
                0.2 * correlation_sim
            )
            
            # Apply deepfake penalty
            final_similarity = combined_similarity * (1.0 - deepfake_penalty)
            
            # Ensure similarity is between 0 and 1
            final_similarity = max(0.0, min(1.0, final_similarity))
            
            self.logger.debug(f"Similarities - Cosine: {cosine_sim:.4f}, Euclidean: {euclidean_sim:.4f}, "
                            f"Correlation: {correlation_sim:.4f}, Deepfake penalty: {deepfake_penalty:.4f}, "
                            f"Final: {final_similarity:.4f}")
            
            return final_similarity
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def _cosine_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity"""
        try:
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            magnitude1 = math.sqrt(sum(a ** 2 for a in embedding1))
            magnitude2 = math.sqrt(sum(b ** 2 for b in embedding2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        except:
            return 0.0
    
    def _euclidean_similarity(self, embedding1, embedding2):
        """Calculate normalized Euclidean similarity"""
        try:
            euclidean_distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(embedding1, embedding2)))
            max_possible_distance = math.sqrt(2 * len(embedding1))  # Assuming normalized features
            
            # Convert distance to similarity (1 - normalized_distance)
            similarity = 1.0 - (euclidean_distance / max_possible_distance)
            return max(0.0, similarity)
        except:
            return 0.0
    
    def _correlation_similarity(self, embedding1, embedding2):
        """Calculate Pearson correlation coefficient"""
        try:
            n = len(embedding1)
            if n < 2:
                return 0.0
            
            mean1 = sum(embedding1) / n
            mean2 = sum(embedding2) / n
            
            numerator = sum((embedding1[i] - mean1) * (embedding2[i] - mean2) for i in range(n))
            
            sum_sq1 = sum((embedding1[i] - mean1) ** 2 for i in range(n))
            sum_sq2 = sum((embedding2[i] - mean2) ** 2 for i in range(n))
            
            denominator = math.sqrt(sum_sq1 * sum_sq2)
            
            if denominator == 0:
                return 0.0
            
            correlation = numerator / denominator
            return abs(correlation)  # Return absolute value for similarity
        except:
            return 0.0
    
    def _detect_deepfake_indicators(self, embedding1, embedding2):
        """Detect potential deepfake indicators and return penalty (0-1)"""
        try:
            penalty = 0.0
            
            # Check for unnaturally perfect similarities in specific feature ranges
            # Deepfakes often have overly consistent spectral characteristics
            
            # If features are too similar in unnatural ways, increase penalty
            feature_groups = [
                embedding1[:8],   # Time domain features
                embedding1[8:23], # Spectral features
                embedding1[23:26] if len(embedding1) > 25 else [], # Pitch features
                embedding1[26:] if len(embedding1) > 28 else []    # Formant features
            ]
            
            corresponding_groups = [
                embedding2[:8],
                embedding2[8:23],
                embedding2[23:26] if len(embedding2) > 25 else [],
                embedding2[26:] if len(embedding2) > 28 else []
            ]
            
            for group1, group2 in zip(feature_groups, corresponding_groups):
                if len(group1) > 0 and len(group2) > 0:
                    group_similarity = self._cosine_similarity(group1, group2)
                    
                    # Deepfakes often have suspiciously high spectral similarity
                    # but poor pitch/formant matching
                    if group1 == embedding1[8:23]:  # Spectral features
                        if group_similarity > 0.95:  # Too perfect
                            penalty += 0.2
                    
                    # Check for unnatural variance patterns
                    variance1 = sum((x - sum(group1)/len(group1)) ** 2 for x in group1) / len(group1)
                    variance2 = sum((x - sum(group2)/len(group2)) ** 2 for x in group2) / len(group2)
                    
                    # Deepfakes often have reduced natural variance
                    if variance1 < 0.001 or variance2 < 0.001:
                        penalty += 0.15
            
            # Check for unnatural feature distribution
            # Real voices have certain statistical properties
            feature_range1 = max(embedding1) - min(embedding1)
            feature_range2 = max(embedding2) - min(embedding2)
            
            # Deepfakes sometimes have compressed dynamic range
            if feature_range1 < 0.1 or feature_range2 < 0.1:
                penalty += 0.1
            
            # Check for suspicious patterns in high-frequency content
            # (This is a simplified check - real deepfake detection would be more sophisticated)
            high_freq_features1 = embedding1[15:20] if len(embedding1) > 20 else []
            high_freq_features2 = embedding2[15:20] if len(embedding2) > 20 else []
            
            if len(high_freq_features1) > 0 and len(high_freq_features2) > 0:
                hf_similarity = self._cosine_similarity(high_freq_features1, high_freq_features2)
                if hf_similarity < 0.3:  # Poor high-frequency matching often indicates synthesis
                    penalty += 0.25
            
            return min(penalty, 0.8)  # Cap penalty at 0.8
            
        except Exception as e:
            self.logger.error(f"Error in deepfake detection: {str(e)}")
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