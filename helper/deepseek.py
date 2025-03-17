import asyncio
import logging
from typing import Any, Optional, Iterator
from typing import AsyncIterator
from typing import Any, Dict, Iterator, List, Optional
from typing_extensions import List, TypedDict

from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, LLMResult
from langchain_core.callbacks import CallbackManagerForLLMRun

from langchain_openai import ChatOpenAI

DEFAULT_API_BASE = "https://api.deepseek.com/v1"

logger = logging.getLogger(__name__)

class DeepseekChatOpenAI(ChatOpenAI):
    def __init__(self, **params: Dict[str, Any]):
        if "base_url" not in params:
            params["base_url"] = DEFAULT_API_BASE
        super().__init__(**params)

    async def _astream(
            self,
            messages: Any,
            stop: Optional[Any] = None,
            run_manager: Optional[Any] = None,
            **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        openai_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")

        params = {
            "model": self.model_name,
            "messages": openai_messages,
            **self.model_kwargs,
            **kwargs,
            "extra_body": {
                "enable_enhanced_generation": True,
                **(kwargs.get("extra_body", {})),
                **(self.model_kwargs.get("extra_body", {}))
            }
        }
        params = {k: v for k, v in params.items() if v not in (None, {}, [])}

        # Create and process the stream
        async for chunk in await self.async_client.create(
                stream=True,
                **params
        ):
            content = chunk.choices[0].delta.content or ""
            reasoning = chunk.choices[0].delta.model_extra.get("reasoning_content", "") if chunk.choices[
                0].delta.model_extra else ""
            if content:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=content)
                )
            if reasoning:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        reasoning_content=reasoning
                    )
                )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        openai_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            else:
                raise ValueError(f"Unsupported message type: {type(msg)}")

        params = {
            "model": self.model_name,
            "messages": openai_messages,
            **self.model_kwargs,
            **kwargs,
            "extra_body": {
                "enable_enhanced_generation": True,
                **(kwargs.get("extra_body", {})),
                **(self.model_kwargs.get("extra_body", {}))
            }
        }
        params = {k: v for k, v in params.items() if v not in (None, {}, [])}

        # Create and process the stream
        for chunk in self.client.create(
                stream=True,
                **params
        ):
            content = chunk.choices[0].delta.content or ""
            reasoning = chunk.choices[0].delta.model_extra.get("reasoning_content", "") if chunk.choices[
                0].delta.model_extra else ""
            if content:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=content)
                )
            if reasoning:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        reasoning_content=reasoning
                    )
                )

    def invoke(
            self,
            messages: Any,
            stop: Optional[Any] = None,
            run_manager: Optional[Any] = None,
            **kwargs: Any,
    ) -> AIMessage:

        async def _ainvoke():
            combined_content = []
            combined_reasoning = []
            async for chunk in self._astream(messages, stop, run_manager, **kwargs):
                if chunk.message.content:
                    combined_content.append(chunk.message.content)
                if "reasoning" in chunk.message.additional_kwargs:
                    combined_reasoning.append(
                        chunk.message.additional_kwargs["reasoning"]
                    )
            return AIMessage(
                content="".join(combined_content),
                reasoning_content="".join(combined_reasoning) if combined_reasoning else ""
            )

        return asyncio.run(_ainvoke())