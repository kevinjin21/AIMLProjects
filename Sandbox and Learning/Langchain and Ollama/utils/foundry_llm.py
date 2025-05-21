from typing import Any, Dict, List, Optional

from pydantic import SecretStr
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

import requests
import json
import time
import re
import os
from dotenv import load_dotenv
load_dotenv('.env')

class FoundryLLM(LLM):
    """Wrapper around Foundry LLM API.

    Args:
        url: The URL of the Foundry LLM API endpoint.
        client_id: Client ID for OAuth2 authentication.
        client_secret: Client secret for OAuth2 authentication.
        key: Optional OAuth2 token for authentication. If not provided, it will be
            generated using the client ID and secret.
    """
    url: str
    client_id: SecretStr
    client_secret: SecretStr
    key: Optional[SecretStr] = None

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string.
        """
        if stop is not None:
            raise ValueError("stop kwargs are not implemented.")
        
        if self.key is None:
            self.key = self.get_token()

        payload = self.generate_payload(prompt)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.key.get_secret_value()}'
        }

        r = requests.post(url=self.url, headers=headers, data=payload)
        data = r.json()
        
        return data['value']

    def get_token(self):
        """
        Obtain an OAuth2 token from the Foundry API using client credentials.
        Retries up to 3 times if the request fails.
        """
        request_body = {
            'grant_type': 'client_credentials',
            'client_id': self.client_id.get_secret_value(),
            'client_secret': self.client_secret.get_secret_value(),
            'scope': 'api:read-data api:write-data compass:edit'
        }
        try:
            max_allowed = 3
            attempt = 0
            r = requests.post("https://apachecorp.palantirfoundry.com/multipass/api/oauth2/token", data=request_body)
            while r.status_code != 200 and attempt < max_allowed:
                attempt += 1
                time.sleep(0.1)
                r = requests.post("https://apachecorp.palantirfoundry.com/multipass/api/oauth2/token", data=request_body)
            self.key = SecretStr(r.json().get('access_token'))
            print("Token generated")
            return self.key
        except Exception as exc:
            raise RuntimeError("Unable to get Foundry token") from exc
        
    def generate_payload(self, prompt: str):
        """
        Helper function to generate payload format.
        Should handle all functions in Developer Console sharing this client id.
        """
        #endpoint = re.search(r"queries/([A-Za-z]+)", self.url).group(1)
        # todo: add formatting for other tools

        print("Payload generated")
        payload = json.dumps({
                    "parameters": {
                        "userInput": prompt
                    }
                })
        return payload

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": "FoundryLLM",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "custom"
    