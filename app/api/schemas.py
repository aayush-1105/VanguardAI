from pydantic import BaseModel


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[dict]
    stream: bool = False
