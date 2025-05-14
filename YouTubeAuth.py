import os
import pickle
import logging
from pathlib import Path
from typing import Tuple, Optional

# Google Auth imports
try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError as e:
    print("Required Google libraries not installed. Please install with:")
    print("pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YouTubeAuth:
    """
    Handles YouTube API authentication and token management
    """
    
    def __init__(self, credentials_file: str = 'client_secrets.json', token_file: str = 'token.pickle'):
        """
        Initialize YouTubeAuth
        
        Args:
            credentials_file: Path to OAuth2 credentials JSON file from Google Cloud Console
            token_file: Path to store/load pickled authentication token
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.credentials = None
        
        # OAuth2 scopes required for the application
        self.SCOPES = [
            'https://www.googleapis.com/auth/youtube.readonly',
            'https://www.googleapis.com/auth/youtube.force-ssl',
            'https://www.googleapis.com/auth/yt-analytics.readonly',
            'https://www.googleapis.com/auth/youtubepartner'
        ]
        
    def _load_credentials(self) -> Optional[Credentials]:
        """
        Load credentials from pickle file if it exists
        
        Returns:
            Credentials object or None if not found/invalid
        """
        if os.path.exists(self.token_file):
            try:
                logger.info(f"Loading credentials from {self.token_file}")
                with open(self.token_file, 'rb') as token:
                    creds = pickle.load(token)
                    if isinstance(creds, Credentials):
                        return creds
                    else:
                        logger.warning("Invalid credentials format in pickle file")
                        return None
            except Exception as e:
                logger.error(f"Error loading credentials: {e}")
                return None
        return None
    
    def _save_credentials(self, creds: Credentials) -> None:
        """
        Save credentials to pickle file
        
        Args:
            creds: Credentials object to save
        """
        try:
            logger.info(f"Saving credentials to {self.token_file}")
            with open(self.token_file, 'wb') as token:
                pickle.dump(creds, token)
            logger.info("Credentials saved successfully")
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
    
    def _perform_oauth_flow(self) -> Credentials:
        """
        Perform OAuth2 flow to get new credentials
        
        Returns:
            New Credentials object
        """
        if not os.path.exists(self.credentials_file):
            raise FileNotFoundError(
                f"Credentials file not found: {self.credentials_file}\n"
                "Please download it from Google Cloud Console:\n"
                "1. Go to https://console.cloud.google.com/\n"
                "2. Create or select a project\n"
                "3. Enable YouTube Data API v3 and YouTube Analytics API\n"
                "4. Create OAuth2 credentials\n"
                "5. Download and save as 'client_secrets.json'"
            )
        
        logger.info("Starting OAuth2 authentication flow")
        flow = InstalledAppFlow.from_client_secrets_file(
            self.credentials_file, 
            self.SCOPES
        )
        
        # Run local server for authentication
        creds = flow.run_local_server(
            port=0,
            success_message='Authentication successful! You can close this window.',
            open_browser=True
        )
        
        return creds
    
    def authenticate(self) -> Tuple[object, object]:
        """
        Main authentication method
        
        Returns:
            Tuple of (youtube_data_service, youtube_analytics_service)
        """
        logger.info("Starting YouTube authentication process")
        
        # Try to load existing credentials
        self.credentials = self._load_credentials()
        
        # Check if credentials are valid
        if not self.credentials or not self.credentials.valid:
            if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                # Refresh expired token
                logger.info("Refreshing expired token")
                try:
                    self.credentials.refresh(Request())
                    self._save_credentials(self.credentials)
                except Exception as e:
                    logger.error(f"Failed to refresh token: {e}")
                    # Perform new OAuth flow
                    self.credentials = self._perform_oauth_flow()
                    self._save_credentials(self.credentials)
            else:
                # Perform new OAuth flow
                self.credentials = self._perform_oauth_flow()
                self._save_credentials(self.credentials)
        
        # Build service objects
        try:
            logger.info("Building YouTube service objects")
            youtube_data = build('youtube', 'v3', credentials=self.credentials)
            youtube_analytics = build('youtubeAnalytics', 'v2', credentials=self.credentials)
            
            # Test the connection
            logger.info("Testing YouTube connection")
            youtube_data.channels().list(part='id', mine=True).execute()
            
            logger.info("Authentication successful!")
            return youtube_data, youtube_analytics
            
        except HttpError as e:
            logger.error(f"API Error: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to build services: {e}")
            raise
    
    def revoke_credentials(self) -> None:
        """
        Revoke stored credentials and delete token file
        """
        if os.path.exists(self.token_file):
            try:
                os.remove(self.token_file)
                logger.info(f"Removed token file: {self.token_file}")
            except Exception as e:
                logger.error(f"Error removing token file: {e}")
        
        self.credentials = None
        logger.info("Credentials revoked")


# Utility function for quick authentication
def quick_auth() -> Tuple[object, object]:
    """
    Quick authentication function for simple use cases
    
    Returns:
        Tuple of (youtube_data_service, youtube_analytics_service)
    """
    auth = YouTubeAuth()
    return auth.authenticate()


# Example usage
if __name__ == "__main__":
    try:
        # Create authentication instance
        auth = YouTubeAuth()
        
        # Authenticate and get service objects
        youtube_data, youtube_analytics = auth.authenticate()
        
        # Test by getting channel info
        request = youtube_data.channels().list(
            part="snippet,contentDetails,statistics",
            mine=True
        )
        response = request.execute()
        
        if response.get('items'):
            channel = response['items'][0]
            print(f"Authenticated as: {channel['snippet']['title']}")
            print(f"Subscriber count: {channel['statistics']['subscriberCount']}")
        else:
            print("No channel found for authenticated user")
            
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise