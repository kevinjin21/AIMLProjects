"""LLM client wrapper supporting multiple providers (Anthropic, Ollama)."""

import os
from typing import List, Optional, Literal


ProviderType = Literal["anthropic", "ollama"]


class LLMClient:
    """Flexible LLM client that can use Anthropic API or local Ollama.
    
    Usage:
        # Via environment variable
        client = LLMClient()  # reads LLM_PROVIDER from .env
        
        # Explicit provider
        client = LLMClient(provider="anthropic", model_name="claude-3-5-sonnet-20241022")
        client = LLMClient(provider="ollama", model_name="llama2")
        
        # Generate response
        messages = [{"role": "user", "content": "Hello!"}]
        response = client.generate(messages)
    """
    
    def __init__(
        self,
        provider: Optional[ProviderType] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Initialize the LLM client.
        
        Args:
            provider: "anthropic" or "ollama". If None, reads from LLM_PROVIDER env var
            model_name: Model to use. If None, uses defaults based on provider
            api_key: API key for Anthropic. If None, reads from ANTHROPIC_API_KEY env var
            base_url: Base URL for Ollama. If None, uses http://localhost:11434
        """
        # Determine provider
        self.provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
        
        if self.provider not in ["anthropic", "ollama"]:
            raise ValueError(f"Unsupported provider: {self.provider}. Use 'anthropic' or 'ollama'.")
        
        # Set model defaults
        if model_name is None:
            if self.provider == "anthropic":
                self.model_name = "claude-haiku-4-5"
            else:  # ollama
                self.model_name = "llama3:8b"
        else:
            self.model_name = model_name
        
        # Initialize the appropriate client
        if self.provider == "anthropic":
            self._init_anthropic(api_key)
        else:
            self._init_ollama(base_url)
    
    def _init_anthropic(self, api_key: Optional[str]):
        """Initialize Anthropic client."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )
        
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def _init_ollama(self, base_url: Optional[str]):
        """Initialize Ollama client."""
        try:
            import ollama
        except ImportError:
            raise ImportError(
                "ollama package not installed. Install with: pip install ollama"
            )
        
        # Ollama client can work with default localhost or custom base_url
        if base_url:
            self.client = ollama.Client(host=base_url)
        else:
            self.client = ollama.Client()  # uses default http://localhost:11434
    
    def generate(self, messages: List[dict], max_tokens: int = 1024) -> str:
        """Generate a response from the LLM.
        
        Args:
            messages: List of message dicts with "role" and "content" keys
                     Roles: "system", "user", "assistant"
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response as a string
        """
        if self.provider == "anthropic":
            return self._generate_anthropic(messages, max_tokens)
        else:
            return self._generate_ollama(messages)
    
    def _generate_anthropic(self, messages: List[dict], max_tokens: int) -> str:
        """Generate response using Anthropic API."""
        # Anthropic expects system message separately
        system_message = None
        filtered_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                filtered_messages.append(msg)
        
        # Call Anthropic API
        kwargs = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": filtered_messages,
        }
        
        if system_message:
            kwargs["system"] = system_message
        
        response = self.client.messages.create(**kwargs)
        return response.content[0].text
    
    def _generate_ollama(self, messages: List[dict]) -> str:
        """Generate response using Ollama."""
        response = self.client.chat(
            model=self.model_name,
            messages=messages,
        )
        return response["message"]["content"]
