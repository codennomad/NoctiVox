import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the Auth directory to the Python path if needed
current_dir = Path(__file__).parent
if current_dir.name != "Auth":
    auth_dir = current_dir / "Auth"
    if auth_dir.exists():
        sys.path.append(str(auth_dir))

# Import YouTubeAuth
try:
    from YouTubeAuth import YouTubeAuth
except ImportError:
    logger.error("YouTubeAuth module not found. Please ensure it's in the correct location.")
    raise


def authenticate_youtube() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Run YouTube Authentication process.
    
    Returns:
        Tuple of (youtube_data, youtube_analytics) or (None, None) if authentication fails
    """
    logger.info("**** Authentication Process Started ****")
    
    logger.info("=== YouTube Authentication ===")
    try:
        youtube_auth = YouTubeAuth()
        youtube_data, youtube_analytics = youtube_auth.authenticate()
        
        if youtube_data and youtube_analytics:
            logger.info("YouTube authentication successful!")
            logger.info("YouTube token has been pickled and can be reused in future sessions.")
            
            # Check for pickle file
            token_path = current_dir / 'token.pickle'
            if token_path.exists():
                logger.info(f"Token file found at: {token_path}")
            
            return youtube_data, youtube_analytics
        else:
            logger.error("YouTube authentication process failed. Please check your credentials.")
            return None, None
            
    except Exception as e:
        logger.error(f"Error during YouTube authentication: {str(e)}")
        raise
    finally:
        logger.info("=== Authentication Process Completed ===")


def main() -> bool:
    """
    Main function to run authentication.
    This is what NoctiVox2 expects to import.
    
    Returns:
        bool: True if authentication successful, False otherwise
    """
    try:
        youtube_data, youtube_analytics = authenticate_youtube()
        return youtube_data is not None and youtube_analytics is not None
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        return False


if __name__ == "__main__":
    # Run authentication when script is executed directly
    success = main()
    sys.exit(0 if success else 1)