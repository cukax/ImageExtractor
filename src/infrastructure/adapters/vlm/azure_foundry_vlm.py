"""Microsoft AI Foundry vision adapter, authenticated with Microsoft Entra ID.

The enterprise variant of the OpenAI adapter. Two things differ:

* The endpoint is an Azure AI Foundry / Azure OpenAI resource, addressed by
  *deployment* name rather than by model name.
* Authentication is secretless by default. ``DefaultAzureCredential`` resolves a
  Managed Identity on a deployed host, a workload identity in AKS, or the
  developer's ``az login`` session locally, so no API key is ever stored, rotated
  or leaked. A key is accepted only when one is explicitly configured, which is
  the escape hatch for environments where Entra ID is not available.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

from ....ports.vlm_port import VLMProviderError
from .base import BaseVLMAdapter, build_data_uri

LOGGER = logging.getLogger(__name__)

#: Token audience for Azure AI services. This is the scope a Managed Identity
#: must be granted (role "Cognitive Services OpenAI User" or higher) against the
#: Foundry resource.
DEFAULT_CREDENTIAL_SCOPE = "https://cognitiveservices.azure.com/.default"

#: API version exposing the vision content parts used below.
DEFAULT_API_VERSION = "2024-10-21"


class AzureAIFoundryVLMAdapter(BaseVLMAdapter):
    """Implements VLMProviderPort on top of a Microsoft AI Foundry deployment."""

    def __init__(
        self,
        endpoint: str,
        deployment: str,
        *,
        api_key: Optional[str] = None,
        api_version: str = DEFAULT_API_VERSION,
        credential_scope: str = DEFAULT_CREDENTIAL_SCOPE,
        classifier_deployment: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        timeout_seconds: float = 60.0,
        image_detail: str = "high",
    ) -> None:
        super().__init__(
            model=deployment,
            provider_name="azure_foundry",
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_seconds=timeout_seconds,
            classifier_model=classifier_deployment,
        )
        if not endpoint:
            raise VLMProviderError(
                self.provider_name,
                "AZURE_FOUNDRY_ENDPOINT is required.",
            )
        if not deployment:
            raise VLMProviderError(
                self.provider_name,
                "AZURE_FOUNDRY_DEPLOYMENT is required.",
            )
        self._endpoint = endpoint
        self._api_key = api_key
        self._api_version = api_version
        self._credential_scope = credential_scope
        # "high" forces the tiled high resolution path, which is what makes a
        # magnified ROI crop worth sending in the first place.
        self.image_detail = image_detail
        self._client: Any = None
        self._credential: Any = None

    @property
    def uses_managed_identity(self) -> bool:
        """True when the adapter authenticates through Entra ID rather than a key."""
        return not self._api_key

    # -- Client management -------------------------------------------------- #
    @property
    def client(self) -> Any:
        """Lazily instantiate and cache the Azure OpenAI client."""
        if self._client is None:
            try:
                from openai import AzureOpenAI
            except ImportError as error:  # pragma: no cover - environment dependent
                raise VLMProviderError(
                    self.provider_name,
                    "The openai package is not installed. Run: pip install openai",
                ) from error

            common: dict[str, Any] = {
                "azure_endpoint": self._endpoint,
                "api_version": self._api_version,
                "timeout": self.timeout_seconds,
            }
            if self._api_key:
                LOGGER.warning(
                    "Authenticating to AI Foundry with an API key; prefer a Managed "
                    "Identity by leaving AZURE_FOUNDRY_API_KEY empty."
                )
                self._client = AzureOpenAI(api_key=self._api_key, **common)
            else:
                self._client = AzureOpenAI(
                    azure_ad_token_provider=self._token_provider(),
                    **common,
                )
        return self._client

    def _token_provider(self) -> Any:
        """Build the Entra ID bearer token provider.

        The provider is a callable the SDK invokes per request, so token refresh
        is handled by azure-identity rather than by this adapter.
        """
        try:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        except ImportError as error:  # pragma: no cover - environment dependent
            raise VLMProviderError(
                self.provider_name,
                "The azure-identity package is not installed. "
                "Run: pip install azure-identity",
            ) from error

        self._credential = DefaultAzureCredential()
        LOGGER.info(
            "AI Foundry authentication resolved through DefaultAzureCredential (scope %s)",
            self._credential_scope,
        )
        return get_bearer_token_provider(self._credential, self._credential_scope)

    # -- Adapter hook ------------------------------------------------------- #
    def _invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        images: Sequence[bytes],
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """Send one multimodal chat completion request carrying every page."""
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        content.extend(
            {
                "type": "image_url",
                "image_url": {
                    "url": build_data_uri(image_bytes),
                    "detail": self.image_detail,
                },
            }
            for image_bytes in images
        )

        try:
            response = self.client.chat.completions.create(
                # On Azure the "model" argument names the deployment, not the
                # underlying model.
                model=model or self.model,
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
            )
        except Exception as error:  # noqa: BLE001 - SDK exceptions are wrapped by design
            raise VLMProviderError(self.provider_name, str(error)) from error

        choices = getattr(response, "choices", None)
        if not choices:
            raise VLMProviderError(self.provider_name, "The response contained no choices.")
        message_content = choices[0].message.content
        if message_content is None:
            raise VLMProviderError(self.provider_name, "The response message had no content.")
        return str(message_content)

    def close(self) -> None:
        """Close the SDK client and the credential it borrowed."""
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()
        self._client = None
        if self._credential is not None and hasattr(self._credential, "close"):
            self._credential.close()
        self._credential = None
