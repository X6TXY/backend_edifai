import io

from app.utils import AppModel
from fastapi import Depends
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import Field

from ..repository.repository import StoryBotRepository
from ..service import Service, get_service
from . import router


class GetAnswerRequest(AppModel):
    request: str


class GetAnswerResponse(AppModel):
    audio: str


@router.get("/generate_story", response_class=StreamingResponse)
def generate_story(
    request: str = "",
    svc: Service = Depends(get_service),
) -> StreamingResponse:
    story_text = svc.repository.generate_story(request)
    audio_bytes = svc.repository.generate_audio(
        story_text, "Bella"
    )  # Replace "Bella" with the desired voice name

    # Create an in-memory stream to serve the audio bytes
    audio_stream = io.BytesIO(audio_bytes)

    # Return the audio as a streaming response
    return StreamingResponse(audio_stream, media_type="audio/mpeg")
