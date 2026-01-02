from pydantic import BaseModel

class RetrievalResponse(BaseModel):
    clipped_text: str