import os 
import sys
import asyncio
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
from functools import lru_cache
from Auth import main as auth_main
from STT import VideoSTT, STTModels
from LLM import create_client, process_text, load_system_prompt
from collections import deque
from dataclasses import dataclass
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('noctivox.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class VideoProcessingResult:
    """Video processing result data class"""
    video_path: Path
    transcription_path: Path
    new_title: Optional[str]
    processing_time: float
    error: Optional[str] = None

class NoctiVoxOptimized:
    def __init__(self, 
                 max_workers: int = 4,
                 cache_dir: str = ".noctivox_cache",
                 enable_cache: bool = True):
        """
        Initialize optimized NoctiVox.
        
        Args:
            max_workers: Maximum number of workers for parallel processing
            cache_dir: Directory for transcription cache
            enable_cache: Enable transcription caching
        """
        self.max_workers = max_workers
        self.cache_dir = Path(cache_dir)
        self.enable_cache = enable_cache
        self.mp4_queue = deque()
        self.stt_models = None
        self.llm_client = None
        self.system_prompt = None
        
        # Create cache directory if needed
        if self.enable_cache:
            self.cache_dir.mkdir(exist_ok=True)
    
    def run_authentication(self):
        """Run the authentication process"""
        logger.info("Starting authentication process...")
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(current_dir)
        
        auth_main()
        logger.info("Authentication process completed.")
    
    def detect_video_files(self) -> List[Path]:
        """Detect video files in the directory"""
        logger.info("Detecting video files...")
        
        current_dir = Path.cwd()
        video_files = []
        
        # Formatos de vídeo suportados
        video_formats = ['*.mp4', '*.mkv', '*.avi', '*.mov', '*.wmv', '*.flv', '*.webm']
        
        for format in video_formats:
            files = list(current_dir.glob(format))
            video_files.extend(files)
        
        if video_files:
            logger.info(f"Found {len(video_files)} video file(s):")
            for file in video_files:
                logger.info(f" - {file.name}")
                self.mp4_queue.append(file)
        else:
            logger.info("No video files found.")
                
        return video_files
    
    def get_file_hash(self, file_path: Path) -> str:
        """Generate unique hash for file"""
        with open(file_path, 'rb') as f:
            # Read only first 1MB for hash (faster)
            file_hash = hashlib.md5(f.read(1024 * 1024)).hexdigest()
        return file_hash
    
    def get_cache_path(self, video_path: Path) -> Path:
        """Return cache path for a video"""
        file_hash = self.get_file_hash(video_path)
        return self.cache_dir / f"{file_hash}.pkl"
    
    def load_from_cache(self, video_path: Path) -> Optional[Dict[str, Any]]:
        """Load transcription from cache if exists"""
        if not self.enable_cache:
            return None
            
        cache_path = self.get_cache_path(video_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    logger.info(f"Loading from cache: {video_path.name}")
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
        return None
    
    def save_to_cache(self, video_path: Path, data: Dict[str, Any]):
        """Save transcription to cache"""
        if not self.enable_cache:
            return
            
        cache_path = self.get_cache_path(video_path)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved to cache: {video_path.name}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def initialize_models(self):
        """Initialize models once"""
        logger.info("Initializing models...")
        
        # STT Models
        self.stt_models = STTModels(
            whisper_model_size="base",
            enable_diarization=True
        )
        
        # LLM Client
        api_key = os.environ.get("OPENROUTER_KEY")
        if api_key:
            self.llm_client = create_client(api_key)
            system_prompt_path = Path(__file__).parent / "system_prompt.json"
            self.system_prompt = load_system_prompt(str(system_prompt_path))
        else:
            logger.warning("OPENROUTER_KEY not found. Title generation disabled.")
    
    def create_enhanced_title_prompt(self, transcription_text: str, video_info: Dict[str, Any]) -> str:
        """
        Create an enhanced prompt for better title generation.
        
        Args:
            transcription_text: Text from video transcription
            video_info: Additional video information
        
        Returns:
            Enhanced prompt for title generation
        """
        # Extract key information
        duration = video_info.get('duration', 'unknown')
        speakers_count = video_info.get('speakers_count', 1)
        
        # Create a comprehensive prompt
        prompt = f"""You are an expert content creator tasked with generating a perfect video title.

VIDEO INFORMATION:
- Duration: {duration}
- Number of speakers: {speakers_count}
- Transcription excerpt: "{transcription_text[:800]}"

TITLE REQUIREMENTS:
1. **Engaging**: Must capture viewer attention immediately
2. **SEO-Optimized**: Include relevant keywords naturally
3. **Concise**: Maximum 60 characters for optimal display
4. **Descriptive**: Clearly convey the video's main topic
5. **Action-Oriented**: Use strong verbs when appropriate
6. **Emotion-Driven**: Evoke curiosity, urgency, or interest

FORMATTING RULES:
- Capitalize first letter of each major word
- No all-caps words unless acronyms
- No excessive punctuation (max 1 punctuation mark)
- No clickbait or misleading content
- No generic phrases like "Must Watch" or "Amazing"

ANALYSIS STEPS:
1. Identify the main topic/theme
2. Extract key points or takeaways
3. Determine the target audience
4. Find the most compelling angle
5. Craft a title that maximizes click-through rate

Based on this analysis, generate ONE perfect title that would make viewers want to watch this video immediately.

TITLE:"""
        
        return prompt
    
    def process_single_video(self, video_path: Path) -> VideoProcessingResult:
        """Process a single video"""
        start_time = datetime.now()
        logger.info(f"Processing: {video_path.name}")
        
        try:
            # Check cache
            cached_data = self.load_from_cache(video_path)
            if cached_data:
                logger.info(f"Using cached data for: {video_path.name}")
                return VideoProcessingResult(
                    video_path=video_path,
                    transcription_path=cached_data['transcription_path'],
                    new_title=cached_data.get('new_title'),
                    processing_time=0,
                    error=None
                )
            
            # Create VideoSTT instance
            video_stt = VideoSTT(
                str(video_path),
                stt_models=self.stt_models,
                language=None
            )
            
            # Process audio
            result = video_stt.process()
            
            if not result:
                raise ValueError("No transcription generated")
            
            transcriptions = result['transcription']
            
            # Save individual transcription
            output_path = video_path.with_name(video_path.stem+'_transcription.txt')
            video_stt.save_results(str(output_path))
            
            # Generate title with LLM
            new_title = None
            if self.llm_client:
                try:
                    # Prepare video information
                    video_info = {
                        'duration': self._format_duration(len(transcriptions)),
                        'speakers_count': len(set(s.get('speaker', 'Speaker') for s in transcriptions))
                    }
                    
                    # Extract meaningful content from transcription
                    transcription_text = " ".join([s['text'] for s in transcriptions[:20]])
                    
                    # Create enhanced prompt
                    prompt = self.create_enhanced_title_prompt(transcription_text, video_info)
                    
                    # Generate title
                    new_title = process_text(
                        self.llm_client,
                        "openai/gpt-4",  # Use GPT-4 for better titles
                        prompt,
                        self.system_prompt
                    )
                    
                    # Clean and format title
                    new_title = new_title.strip().replace('"', '').replace("'", '')
                    new_title = new_title[:60]  # Limit length
                    
                    # Sanitize for filename
                    safe_title = "".join(c for c in new_title if c.isalnum() or c in (' ', '-', '_', '.'))
                    safe_title = safe_title.strip()
                    
                    # Rename video
                    new_path = video_path.with_name(f"{safe_title}{video_path.suffix}")
                    
                    if not new_path.exists():
                        os.rename(str(video_path), str(new_path))
                        logger.info(f"Video renamed to: {safe_title}")
                    
                except Exception as e:
                    logger.error(f"Error generating title: {e}")
            
            # Save to cache
            cache_data = {
                'transcriptions': transcriptions,
                'transcription_path': output_path,
                'new_title': new_title,
                'timestamp': datetime.now()
            }
            self.save_to_cache(video_path, cache_data)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return VideoProcessingResult(
                video_path=video_path,
                transcription_path=output_path,
                new_title=new_title,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error processing {video_path.name}: {e}")
            return VideoProcessingResult(
                video_path=video_path,
                transcription_path=None,
                new_title=None,
                processing_time=(datetime.now() - start_time).total_seconds(),
                error=str(e)
            )
    
    def _format_duration(self, segments_count: int) -> str:
        """Format duration based on segment count (approximation)"""
        # Approximate: each segment ~3 seconds
        total_seconds = segments_count * 3
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}:{seconds:02d}"
    
    async def process_videos_async(self):
        """Process videos asynchronously"""
        videos = list(self.mp4_queue)
        
        if not videos:
            logger.info("No videos to process.")
            return
        
        # Initialize models
        self.initialize_models()
        
        # Process in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for video in videos:
                future = executor.submit(self.process_single_video, video)
                futures.append(future)
            
            # Wait for completion and collect results
            results = []
            for future in futures:
                result = future.result()
                results.append(result)
                
                if result.error:
                    logger.error(f"Error in {result.video_path.name}: {result.error}")
                else:
                    logger.info(f"Completed {result.video_path.name} in {result.processing_time:.2f}s")
        
        # Generate final report
        self.generate_report(results)
    
    def generate_report(self, results: List[VideoProcessingResult]):
        """Generate processing report"""
        logger.info("Generating final report...")
        
        report_path = Path("processing_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=== NoctiVox Processing Report ===\n\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Total videos: {len(results)}\n")
            
            successful = [r for r in results if not r.error]
            failed = [r for r in results if r.error]
            
            f.write(f"Successful: {len(successful)}\n")
            f.write(f"Failed: {len(failed)}\n\n")
            
            if successful:
                f.write("=== Successfully Processed Videos ===\n")
                for result in successful:
                    f.write(f"\n- {result.video_path.name}\n")
                    f.write(f"  Time: {result.processing_time:.2f}s\n")
                    if result.new_title:
                        f.write(f"  New title: {result.new_title}\n")
            
            if failed:
                f.write("\n=== Failures ===\n")
                for result in failed:
                    f.write(f"\n- {result.video_path.name}\n")
                    f.write(f"  Error: {result.error}\n")
        
        logger.info(f"Report saved to: {report_path}")
    
    async def run(self):
        """Run the complete pipeline"""
        logger.info("=== NoctiVox Optimized Starting ===")
        
        # Authentication
        self.run_authentication()
        
        # Detect MP4s
        self.detect_video_files()
        
        # Process videos
        await self.process_videos_async()
        
        logger.info("=== NoctiVox Optimized Completed ===")

def main():
    """Main function"""
    noctivox = NoctiVoxOptimized(
        max_workers=4,  # Adjust based on hardware
        enable_cache=True
    )
    
    # Run asynchronously
    asyncio.run(noctivox.run())

if __name__ == "__main__":
    main()
