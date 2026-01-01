from pydantic import BaseModel

class ClipRequest(BaseModel):
    text: str