import os
import sys
import json
import logging
import tempfile
import warnings
import time
import datetime
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from collections import deque

import torch
import numpy as np
import faster_whisper
from scipy.io import wavfile
from moviepy.editor import VideoFileClip
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pydub import AudioSegment
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
def setup_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with console and file handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"stt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logger("VideoSTT", logging.INFO)

# Check for diarization availability
try:
    from pyannote.audio import Pipeline
    DIARIZATION_AVAILABLE = True
    logger.info("Speaker diarization is available")
except ImportError:
    DIARIZATION_AVAILABLE = False
    logger.warning("pyannote.audio not installed. Speaker diarization will not be available.")
    logger.warning("To enable speaker diarization, install with: pip install pyannote.audio")

# Hugging Face token for pyannote.audio
HF_TOKEN = os.getenv("HF_TOKEN", "")


class STTModels:
    """
    Class to initialize and hold speech-to-text models
    This allows models to be initialized and reused for multiple videos
    """
    
    def __init__(self, 
                 whisper_model_size: str = "large", 
                 enable_diarization: bool = True, 
                 hf_token: str = HF_TOKEN):
        
        self.whisper_model_size = whisper_model_size
        self.enable_diarization = enable_diarization and DIARIZATION_AVAILABLE
        self.hf_token = hf_token or HF_TOKEN
        
        # Initialize Whisper model
        self.whisper_model = self._initialize_whisper()
        
        # Initialize diarization model if available
        self.diarization_pipeline = self._initialize_diarization()
        
    def _initialize_whisper(self) -> faster_whisper.WhisperModel:
        """Initialize the Whisper model"""
        logger.info(f"Initializing Whisper model (size: {self.whisper_model_size})...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        
        try:
            model = faster_whisper.WhisperModel(
                self.whisper_model_size,
                device=device,
                compute_type=compute_type
            )
            logger.info(f"Whisper model initialized on {device} with compute type {compute_type}")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            raise
            
    def _initialize_diarization(self) -> Optional[Pipeline]:
        """Initialize the diarization pipeline"""
        if not self.enable_diarization:
            logger.info("Speaker diarization disabled")
            return None
            
        if not self.hf_token:
            logger.warning("No Hugging Face token provided. Speaker diarization will be disabled.")
            self.enable_diarization = False
            return None
            
        try:
            logger.info("Initializing speaker diarization model...")
            
            # Set environment variable for HuggingFace auth token
            os.environ["HF_TOKEN"] = self.hf_token
            
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.0",
                use_auth_token=self.hf_token
            )
            
            if torch.cuda.is_available():
                pipeline = pipeline.to(torch.device("cuda"))
                
            logger.info("Speaker diarization model loaded successfully!")
            return pipeline
            
        except Exception as e:
            logger.error(f"Error loading diarization model: {e}")
            logger.error("\nIMPORTANT: You need to accept the user conditions for BOTH models:")
            logger.error("1. https://hf.co/pyannote/speaker-diarization-3.0")
            logger.error("2. https://hf.co/pyannote/segmentation-3.0")
            logger.error("\nVisit both links and click 'Accept' to agree to the terms of use.")
            self.enable_diarization = False
            return None


class VideoSTT:
    """
    Main class for video speech-to-text processing
    """
    
    def __init__(self, 
                 video_path: str, 
                 stt_models: Optional[STTModels] = None,
                 model_size: str = "large",
                 enable_diarization: bool = True, 
                 hf_token: str = HF_TOKEN,
                 language: Optional[str] = None,
                 chunk_size: int = 30):
        
        # Configuration
        self.video_path = Path(video_path)
        self.chunk_size = chunk_size  # Process audio in chunks (seconds)
        self.transcription = []
        self.speakers = {}  # Dictionary to store speaker segments
        self.language = language  # Language for transcription (None for auto-detection)
        
        # Validate video path
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
            
        logger.info(f"Initializing VideoSTT for: {self.video_path.name}")
        
        # Use provided models or initialize new ones
        if stt_models is not None:
            self.model = stt_models.whisper_model
            self.diarization_pipeline = stt_models.diarization_pipeline
            self.enable_diarization = enable_diarization and stt_models.enable_diarization
            logger.info("Using provided STT models")
        else:
            logger.warning("Initializing new models for each video is inefficient.")
            logger.warning("Consider using STTModels class for multiple videos.")
            
            # Initialize models inline
            self._initialize_models(model_size, enable_diarization, hf_token)
            
        # Create temp directory for processing
        self.temp_dir = tempfile.mkdtemp(prefix="videostt_")
        logger.debug(f"Created temp directory: {self.temp_dir}")
        
    def _initialize_models(self, model_size: str, enable_diarization: bool, hf_token: str):
        """Initialize models inline (legacy mode)"""
        # Initialize Whisper model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "int8"
        
        self.model = faster_whisper.WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        
        # Initialize diarization model if available
        self.enable_diarization = enable_diarization and DIARIZATION_AVAILABLE
        self.diarization_pipeline = None
        
        if self.enable_diarization and hf_token:
            try:
                logger.info("Initializing speaker diarization model...")
                
                # Set environment variable for HuggingFace auth token
                os.environ["HF_TOKEN"] = hf_token
                
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.0",
                    use_auth_token=hf_token
                )
                
                if torch.cuda.is_available():
                    self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))
                    
                logger.info("Speaker diarization model loaded successfully!")
                
            except Exception as e:
                logger.error(f"Error loading diarization model: {e}")
                self.enable_diarization = False
                self.diarization_pipeline = None
                
    def extract_audio(self, output_path: Optional[str] = None) -> str:
        """Extract audio from video"""
        if output_path is None:
            output_path = os.path.join(self.temp_dir, "audio.wav")
            
        logger.info("Extracting audio from video...")
        
        try:
            video = VideoFileClip(str(self.video_path))
            audio = video.audio
            
            # Ensure 16kHz sample rate for optimal Whisper performance
            audio.write_audiofile(
                output_path, 
                codec='pcm_s16le',
                fps=16000,
                logger=None  # Suppress moviepy logs
            )
            
            video.close()
            
            logger.info(f"Audio extracted successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            raise
            
    def perform_diarization(self, 
                           audio_path: str, 
                           min_speakers: Optional[int] = None, 
                           max_speakers: Optional[int] = None) -> Optional[List[Dict]]:
        """Perform speaker diarization on the audio file"""
        if not self.enable_diarization:
            logger.info("Speaker diarization is disabled")
            return None
            
        try:
            logger.info("Performing speaker diarization...")
            
            # Diarization options
            diarization_options = {}
            if min_speakers is not None:
                diarization_options["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarization_options["max_speakers"] = max_speakers
                
            # Set up progress monitoring
            try:
                with ProgressHook() as hook:
                    diarization = self.diarization_pipeline(
                        {"audio": audio_path},
                        hook=hook,
                        **diarization_options
                    )
            except (ImportError, AttributeError):
                # Fall back to standard processing
                diarization = self.diarization_pipeline(
                    {"audio": audio_path},
                    **diarization_options
                )
                
            # Extract speaker segments
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Only include segments longer than 0.5 seconds
                if turn.end - turn.start > 0.5:
                    speaker_segments.append({
                        "speaker": speaker,
                        "start": turn.start,
                        "end": turn.end
                    })
                    
            # Merge close segments from the same speaker
            merged_segments = self._merge_close_segments(speaker_segments)
            
            num_speakers = len(set(segment['speaker'] for segment in merged_segments))
            logger.info(f"Identified {num_speakers} speakers")
            
            # Rename speakers to user-friendly names
            speaker_mapping = {}
            for i, speaker in enumerate(sorted(set(segment['speaker'] for segment in merged_segments))):
                speaker_mapping[speaker] = f"Speaker {i+1}"
                
            # Apply the mapping
            for segment in merged_segments:
                segment['speaker'] = speaker_mapping[segment['speaker']]
                
            return merged_segments
            
        except Exception as e:
            logger.error(f"Error during speaker diarization: {e}")
            return None
            
    def _merge_close_segments(self, segments: List[Dict], threshold: float = 0.5) -> List[Dict]:
        """Merge segments from the same speaker that are close together"""
        if not segments:
            return segments
            
        # Sort segments by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])
        
        merged = []
        current = sorted_segments[0].copy()
        
        for segment in sorted_segments[1:]:
            # If same speaker and close enough, merge
            if (segment['speaker'] == current['speaker'] and 
                segment['start'] - current['end'] <= threshold):
                current['end'] = segment['end']
            else:
                merged.append(current)
                current = segment.copy()
                
        merged.append(current)
        
        return merged
        
    def transcribe_audio(self, audio_path: str) -> List[Dict]:
        """Transcribe audio using Whisper"""
        logger.info("Starting audio transcription...")
        
        try:
            # Transcribe with word-level timestamps
            segments, info = self.model.transcribe(
                audio_path,
                language=self.language,
                word_timestamps=True,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.6,
                    min_speech_duration_ms=250,
                    max_speech_duration_s=float('inf'),
                    min_silence_duration_ms=2000,
                    speech_pad_ms=400
                )
            )
            
            # Detect language if not specified
            if self.language is None:
                self.language = info.language
                logger.info(f"Detected language: {self.language}")
                
            # Process segments
            transcription_segments = []
            for segment in tqdm(segments, desc="Processing segments"):
                transcription_segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "words": [
                        {
                            "start": word.start,
                            "end": word.end,
                            "word": word.word
                        }
                        for word in segment.words
                    ] if segment.words else []
                })
                
            logger.info(f"Transcription completed: {len(transcription_segments)} segments")
            return transcription_segments
            
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise
            
    def assign_speakers_to_segments(self, 
                                   transcription_segments: List[Dict], 
                                   speaker_segments: Optional[List[Dict]]):
        """Assign speakers to transcription segments based on overlap"""
        if not speaker_segments:
            logger.info("No speaker segments available")
            return
            
        logger.info("Assigning speakers to transcription segments...")
        
        # Check for overlaps
        has_overlap = self._check_overlap(transcription_segments, speaker_segments)
        
        if not has_overlap:
            logger.warning("No overlap found between speaker and transcription segments.")
            logger.warning("Using proximity-based assignment instead.")
            self._assign_speakers_by_proximity(transcription_segments, speaker_segments)
            return
            
        # Group close transcription segments
        grouped_segments = self._group_close_transcription_segments(transcription_segments)
        logger.info(f"Grouped {len(transcription_segments)} segments into {len(grouped_segments)} groups")
        
        # Process each group
        for group in grouped_segments:
            speaker_votes = {}
            
            for trans_segment in group:
                trans_start = trans_segment["start"]
                trans_end = trans_segment["end"]
                
                for speaker_segment in speaker_segments:
                    speaker_start = speaker_segment["start"]
                    speaker_end = speaker_segment["end"]
                    speaker = speaker_segment["speaker"]
                    
                    # Calculate overlap
                    overlap_start = max(trans_start, speaker_start)
                    overlap_end = min(trans_end, speaker_end)
                    overlap_duration = max(0, overlap_end - overlap_start)
                    
                    if overlap_duration > 0:
                        if speaker not in speaker_votes:
                            speaker_votes[speaker] = 0
                        speaker_votes[speaker] += overlap_duration
                        
            # Assign the dominant speaker
            if speaker_votes:
                dominant_speaker = max(speaker_votes.items(), key=lambda x: x[1])[0]
                for trans_segment in group:
                    trans_segment["speaker"] = dominant_speaker
            else:
                for trans_segment in group:
                    trans_segment["speaker"] = "UNKNOWN"
                    
    def _check_overlap(self, 
                      transcription_segments: List[Dict], 
                      speaker_segments: List[Dict]) -> bool:
        """Check if there's overlap between transcription and speaker segments"""
        for trans_segment in transcription_segments:
            trans_start = trans_segment["start"]
            trans_end = trans_segment["end"]
            
            for speaker_segment in speaker_segments:
                speaker_start = speaker_segment["start"]
                speaker_end = speaker_segment["end"]
                
                # Check for overlap
                if max(trans_start, speaker_start) < min(trans_end, speaker_end):
                    return True
                    
        return False
        
    def _group_close_transcription_segments(self, 
                                          transcription_segments: List[Dict], 
                                          max_gap: float = 1.0) -> List[List[Dict]]:
        """Group transcription segments that are close temporally"""
        if not transcription_segments:
            return []
            
        # Sort segments by start time
        sorted_segments = sorted(transcription_segments, key=lambda x: x["start"])
        groups = []
        current_group = [sorted_segments[0]]
        
        for i in range(1, len(sorted_segments)):
            current_segment = sorted_segments[i]
            previous_segment = sorted_segments[i-1]
            
            # Calculate gap
            gap = current_segment["start"] - previous_segment["end"]
            
            if gap <= max_gap:
                current_group.append(current_segment)
            else:
                groups.append(current_group)
                current_group = [current_segment]
                
        # Add the last group
        if current_group:
            groups.append(current_group)
            
        return groups
        
    def _assign_speakers_by_proximity(self, 
                                     transcription_segments: List[Dict], 
                                     speaker_segments: List[Dict]):
        """Assign speakers based on temporal proximity"""
        for trans_segment in transcription_segments:
            trans_start = trans_segment["start"]
            trans_end = trans_segment["end"]
            trans_mid = (trans_start + trans_end) / 2
            
            min_distance = float('inf')
            nearest_speaker = "UNKNOWN"
            
            for speaker_segment in speaker_segments:
                speaker_start = speaker_segment["start"]
                speaker_end = speaker_segment["end"]
                speaker = speaker_segment["speaker"]
                
                # Calculate distance
                if trans_mid < speaker_start:
                    distance = speaker_start - trans_mid
                elif trans_mid > speaker_end:
                    distance = trans_mid - speaker_end
                else:
                    distance = 0  # Inside speaker segment
                    
                if distance < min_distance:
                    min_distance = distance
                    nearest_speaker = speaker
                    
            trans_segment["speaker"] = nearest_speaker
            
    def process(self) -> Dict[str, Any]:
        """Main processing pipeline"""
        logger.info(f"Starting processing for: {self.video_path.name}")
        
        try:
            # Extract audio
            audio_path = self.extract_audio()
            
            # Perform diarization
            speaker_segments = None
            if self.enable_diarization:
                speaker_segments = self.perform_diarization(audio_path)
                
            # Transcribe audio
            transcription_segments = self.transcribe_audio(audio_path)
            
            # Assign speakers to segments
            if speaker_segments:
                self.assign_speakers_to_segments(transcription_segments, speaker_segments)
                
            # Store results
            self.transcription = transcription_segments
            
            # Generate summary
            result = {
                "video_path": str(self.video_path),
                "language": self.language,
                "duration": self._get_audio_duration(audio_path),
                "num_segments": len(transcription_segments),
                "num_speakers": len(set(s.get("speaker", "UNKNOWN") for s in transcription_segments)),
                "transcription": transcription_segments
            }
            
            logger.info("Processing completed successfully!")
            return result
            
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise
        finally:
            # Cleanup
            self.cleanup()
            
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get duration of audio file in seconds"""
        try:
            audio = AudioSegment.from_wav(audio_path)
            return len(audio) / 1000.0
        except Exception:
            return 0.0
            
    def save_results(self, output_path: str, format: str = "json"):
        """Save transcription results"""
        logger.info(f"Saving results to: {output_path}")
        
        try:
            if format == "json":
                self._save_json(output_path)
            elif format == "srt":
                self._save_srt(output_path)
            elif format == "vtt":
                self._save_vtt(output_path)
            elif format == "txt":
                self._save_txt(output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            logger.info(f"Results saved successfully in {format} format")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
            
    def _save_json(self, output_path: str):
        """Save results as JSON"""
        result = {
            "video_path": str(self.video_path),
            "language": self.language,
            "transcription": self.transcription
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
    def _save_srt(self, output_path: str):
        """Save results as SRT subtitle file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(self.transcription, 1):
                start_time = self._format_timestamp(segment["start"])
                end_time = self._format_timestamp(segment["end"])
                
                speaker = segment.get("speaker", "")
                speaker_prefix = f"[{speaker}] " if speaker else ""
                
                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{speaker_prefix}{segment['text']}\n\n")
                
    def _save_vtt(self, output_path: str):
        """Save results as WebVTT subtitle file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            
            for segment in self.transcription:
                start_time = self._format_timestamp(segment["start"], vtt=True)
                end_time = self._format_timestamp(segment["end"], vtt=True)
                
                speaker = segment.get("speaker", "")
                speaker_prefix = f"[{speaker}] " if speaker else ""
                
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{speaker_prefix}{segment['text']}\n\n")
                
    def _save_txt(self, output_path: str):
        """Save results as plain text in the format:
        [HH:MM:SS.ss - HH:MM:SS.ss] Speaker X: line of transcribed text
        """
        def format_timestamp(seconds: float) -> str:
            td = datetime.timedelta(seconds=seconds)
            total_seconds = td.total_seconds()
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            secs = total_seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"

        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in self.transcription:
                start = segment.get("start", 0.0)
                end = segment.get("end", 0.0)
                speaker = segment.get("speaker", "UNKNOWN")
                text = segment.get("text", "")
                start_str = format_timestamp(start)
                end_str = format_timestamp(end)
                f.write(f"[{start_str} - {end_str}] {speaker}: {text}\n")
                
    def _format_timestamp(self, seconds: float, vtt: bool = False) -> str:
        """Format timestamp for subtitle files"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        
        if vtt:
            return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        else:
            return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{int((seconds % 1) * 1000):03d}"
            
    def cleanup(self):
        """Clean up temporary files"""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
            
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()


# Utility functions
def process_video(video_path: str, 
                 output_dir: str = "output",
                 model_size: str = "large",
                 enable_diarization: bool = True,
                 language: Optional[str] = None,
                 formats: List[str] = ["json", "srt"]) -> Dict[str, Any]:
    """Convenience function to process a single video"""
    logger.info(f"Processing video: {video_path}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize models
    models = STTModels(
        whisper_model_size=model_size,
        enable_diarization=enable_diarization
    )
    
    # Process video
    stt = VideoSTT(
        video_path=video_path,
        stt_models=models,
        language=language
    )
    
    result = stt.process()
    
    # Save results in requested formats
    base_name = Path(video_path).stem
    for format in formats:
        output_path = output_dir / f"{base_name}.{format}"
        stt.save_results(str(output_path), format=format)
        
    return result


def batch_process_videos(video_paths: List[str],
                        output_dir: str = "output",
                        model_size: str = "large",
                        enable_diarization: bool = True,
                        language: Optional[str] = None,
                        formats: List[str] = ["json", "srt"]) -> List[Dict[str, Any]]:
    """Process multiple videos efficiently with shared models"""
    logger.info(f"Batch processing {len(video_paths)} videos")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize models once
    models = STTModels(
        whisper_model_size=model_size,
        enable_diarization=enable_diarization
    )
    
    results = []
    
    for i, video_path in enumerate(video_paths, 1):
        logger.info(f"Processing video {i}/{len(video_paths)}: {video_path}")
        
        try:
            # Process video
            stt = VideoSTT(
                video_path=video_path,
                stt_models=models,
                language=language
            )
            
            result = stt.process()
            
            # Save results
            base_name = Path(video_path).stem
            for format in formats:
                output_path = output_dir / f"{base_name}.{format}"
                stt.save_results(str(output_path), format=format)
                
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            results.append({"error": str(e), "video_path": video_path})
            
    logger.info("Batch processing completed")
    return results


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python STT.py <video_path> [output_dir]")
        sys.exit(1)
        
    video_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    result = process_video(
        video_path=video_path,
        output_dir=output_dir,
        enable_diarization=True,
        formats=["json", "srt", "vtt", "txt"]
    )
    
    print(f"Processing completed. Results saved to: {output_dir}")
