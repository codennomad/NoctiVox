import json
import os
import logging
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

#Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelnames)s - %(message)s',
    handlers=[
        logging.FileHandler('llm_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

#Carregar variaveis do arquivo .env
load_dotenv()

def load_system_prompt(prompt_path):
    """Load system prompt form JSON file."""
    try:
        if not os.path.exists(prompt_path):
            logger.warning(f"{prompt_path} not found. Using default prompt!")
            return "You THE BEST!"
        
        with open(prompt_path, 'r') as file:
            data = json.load(file)
            logger.info(f"System prompt loaded from {prompt_path}")
            return data.get("system_prompt", "")
    except Exception as e:
        logger.error(f"Error loading system prompt: {e}")
        return ""
    

def create_client(api_key: str, base_url="https://openrouter.ai/api/v1") -> OpenAI:
    """Create an OpenAi Client."""
    logger.info("Creating OpenRouter client")
    return OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    
    
def process_text(
    client: OpenAI,
    model: str, 
    user_prompt: str, 
    system_prompt: Optional[str]) -> str:
    """Process text with a system prompt and user prompt."""
    try:
        logger.info(f"Processing text with template: {model}")
        messages = []
        
        #Add system message if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
            
        #Add user message with text only
        messages.append({
            "role": "user",
            "content": user_prompt
        })
        
        logger.debug(f"Sending request to api")
        completion = client.chat.completions.create(
            model=model,
            messages=messages
        )
        logger.info("Answer received successfully")
        # Robustly check the response structure
        if (completion is not None and
            hasattr(completion, "choices") and
            isinstance(completion.choices, list) and
            len(completion.choices) > 0 and
            hasattr(completion.choices[0], "message") and
            hasattr(completion.choices[0].message, "content")):
            return completion.choices[0].message.content
        else:
            logger.error(f"Invalid response structure: {completion}")
            return "Error: Received invalid response from the language model API."
    except Exception as e:
        logger.error(f"Error processing text: {e}", exc_info=True)
        return f"Error processing text: {e}"
    
    
def main(
    system_prompt_path: str="system_prompt.json",
    user_prompt: str="How can I help you today?",
    model: str="") -> None:
    """Main function to process text with modular system prompt."""
    #Obter a chave da API do arquivo .env
    api_key = os.environ.get("OPENROUTER_KEY")
    if not api_key:
        logger.error("OPENROUTER_KEY not found in .env file")
        return
    
    #Load system prompt
    system_prompt = load_system_prompt(system_prompt_path)
    client = create_client(api_key)
    result = process_text(client, model, user_prompt, system_prompt)
    
    print("\n--- Answer ---")
    print(result)
    print("--------------\n")
    
if __name__ == "__main__":
    logger.info("Starting LLM application")
    main()
