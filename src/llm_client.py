"""LLM client abstraction for multiple providers."""
import os
import time
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

# Optional imports - only import when needed
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False


class LLMClient(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> tuple[str, Dict[str, int]]:
        """
        Generate completion from messages.

        Returns:
            (response_text, token_usage)
        """
        pass

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate completion from messages (returns dict format).

        Returns:
            Dict with 'content' and 'usage' keys
        """
        text, tokens = self.complete(messages, temperature, max_tokens, **kwargs)
        return {
            "content": text,
            "usage": {
                "prompt_tokens": tokens.get("input", 0),
                "completion_tokens": tokens.get("output", 0)
            }
        }


class OpenAIClient(LLMClient):
    """OpenAI API client."""
    
    def __init__(self, model: str = "gpt-4"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Install with: pip install openai")
        self.model = model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> tuple[str, Dict[str, int]]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        text = response.choices[0].message.content
        tokens = {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens
        }
        return text, tokens


class TogetherAIClient(LLMClient):
    """TogetherAI API client (OpenAI-compatible)."""

    def __init__(self, model: str = "Qwen/Qwen2.5-72B-Instruct-Turbo"):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Install with: pip install openai")
        self.model = model
        # TogetherAI uses OpenAI-compatible API
        self.client = openai.OpenAI(
            api_key=os.getenv("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1"
        )

    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> tuple[str, Dict[str, int]]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        text = response.choices[0].message.content
        tokens = {
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens
        }
        return text, tokens


class AnthropicClient(LLMClient):
    """Anthropic (Claude) API client."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        self.model = model
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> tuple[str, Dict[str, int]]:
        # Convert messages format if needed
        system_msg = None
        formatted_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                formatted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        response = self.client.messages.create(
            model=self.model,
            messages=formatted_messages,
            system=system_msg,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        text = response.content[0].text
        tokens = {
            "input": response.usage.input_tokens,
            "output": response.usage.output_tokens
        }
        return text, tokens


class GoogleClient(LLMClient):
    """
    Google Gemini API client.
    
    Supported models:
    - gemini-2.5-flash (recommended for cost)
    - gemini-2.5-pro (recommended for quality)
    - gemini-2.0-flash-exp (experimental)
    - gemini-1.5-pro
    - gemini-1.5-flash
    """
    
    def __init__(self, model: str = "gemini-2.5-flash", enable_sanitization: bool = False, rate_limit_delay: float = 1.0):
        if not GOOGLE_AVAILABLE:
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
        self.model = model
        # CRITICAL: Disable sanitization by default - it removes important URL context
        # The agent NEEDS to see full URLs to understand which service to navigate to
        self.enable_sanitization = enable_sanitization
        # Rate limiting to avoid hitting API limits
        self.rate_limit_delay = rate_limit_delay
        self.last_call_time = 0
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.client = genai.GenerativeModel(model)
    
    def _sanitize_for_safety(self, content: str) -> str:
        """
        Sanitize content to reduce safety filter triggers.
        
        Removes patterns that commonly trigger safety filters:
        - Full URLs (replace with [URL])
        - IP addresses (replace with [IP])
        - Port numbers (replace with [PORT])
        - Email addresses (replace with [EMAIL])
        - Specific element IDs (replace with [ID])
        """
        if not content or not self.enable_sanitization:
            return content
        
        import re
        
        # Remove full URLs (most common safety trigger)
        content = re.sub(r'https?://[^\s]+', '[URL]', content)
        
        # Remove IP addresses
        content = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '[IP]', content)
        
        # Remove specific port numbers (keep general references)
        content = re.sub(r':\d{4,5}\b', ':[PORT]', content)
        
        # Remove email addresses
        content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', content)
        
        # Remove specific element IDs that might look suspicious
        content = re.sub(r'bid=["\']?\w+["\']?', 'bid=[ID]', content)
        content = re.sub(r'id=["\']?\w+["\']?', 'id=[ID]', content)
        
        # Remove domain names (can look like phishing)
        content = re.sub(r'\b([a-z0-9-]+\.){2,}[a-z]{2,}\b', '[SITE]', content, flags=re.IGNORECASE)
        
        return content
    
    def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        **kwargs
    ) -> tuple[str, Dict[str, int]]:
        # Rate limiting: ensure minimum delay between API calls
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            time.sleep(sleep_time)

        # Convert messages to Gemini format with sanitization
        system_instruction = None
        formatted_messages = []
        
        for msg in messages:
            # Sanitize content to reduce safety filter triggers
            sanitized_content = self._sanitize_for_safety(msg["content"])
            
            if msg["role"] == "system":
                system_instruction = sanitized_content
            else:
                formatted_messages.append({
                    "role": "user" if msg["role"] == "user" else "model",
                    "parts": [sanitized_content]
                })
        
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        # Configure safety settings to be less restrictive
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            },
        ]
        
        # Create model with system instruction if provided
        if system_instruction:
            model = genai.GenerativeModel(
                self.model,
                system_instruction=system_instruction,
                safety_settings=safety_settings
            )
        else:
            model = genai.GenerativeModel(
                self.model,
                safety_settings=safety_settings
            )
        
        try:
            response = model.generate_content(
                formatted_messages,
                generation_config=generation_config
            )
            
            # Handle blocked responses gracefully
            if not response.candidates:
                return "Error: Response was blocked by safety filters", {"input": 0, "output": 0}
            
            candidate = response.candidates[0]
            
            # Check finish reason
            if candidate.finish_reason == 2:  # SAFETY
                return "Error: Response blocked by safety filters", {"input": 0, "output": 0}
            elif candidate.finish_reason == 3:  # RECITATION
                return "Error: Response blocked due to recitation", {"input": 0, "output": 0}
            
            # Try to get text
            if hasattr(response, 'text'):
                text = response.text
            elif candidate.content and candidate.content.parts:
                text = candidate.content.parts[0].text
            else:
                return "Error: No text in response", {"input": 0, "output": 0}
            
            # Get token counts
            tokens = {
                "input": getattr(response.usage_metadata, "prompt_token_count", 0),
                "output": getattr(response.usage_metadata, "candidates_token_count", 0)
            }

            # Update last call time for rate limiting
            self.last_call_time = time.time()
            return text, tokens

        except Exception as e:
            # Log the error and return gracefully
            import traceback
            error_msg = f"Gemini API error: {str(e)}"
            print(f"Warning: {error_msg}")
            traceback.print_exc()
            # Update last call time even on error
            self.last_call_time = time.time()
            return f"Error: {error_msg}", {"input": 0, "output": 0}


def create_llm_client(provider: str, model: str, rate_limit_delay: float = 1.0) -> LLMClient:
    """
    Factory function to create LLM client.

    Args:
        provider: LLM provider (openai, anthropic, google, together)
        model: Model name
        rate_limit_delay: Minimum delay in seconds between API calls (only for Google)
    """

    clients = {
        "openai": OpenAIClient,
        "anthropic": AnthropicClient,
        "google": GoogleClient,
        "together": TogetherAIClient,
    }

    if provider not in clients:
        raise ValueError(f"Unknown provider: {provider}. Choose from {list(clients.keys())}")

    # Pass rate_limit_delay only to GoogleClient
    if provider == "google":
        return clients[provider](model, rate_limit_delay=rate_limit_delay)
    else:
        return clients[provider](model)
