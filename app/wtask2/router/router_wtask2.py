import logging

from app.auth.adapters.jwt_service import JWTData
from app.auth.router.dependencies import parse_jwt_user_data
from app.utils import AppModel
from fastapi import Depends, HTTPException, Response, status
from pydantic import Field, ValidationError

from ..repository.repository import Wtask2Repository
from ..service import Service, get_service
from . import router

repository = Wtask2Repository()


class GetAnswerRequest(AppModel):
    request: str


class GetAnswerResponse(AppModel):
    response: str


@router.post("/get_answer", response_model=GetAnswerResponse)
def get_answer(
    request: GetAnswerRequest,
    svc: Service = Depends(get_service),
) -> GetAnswerResponse:
    try:
        if not svc.repository.vector_store:
            svc.repository.load_vector_store()

        response = svc.repository.get_answer(request.request)
        return GetAnswerResponse(response=response)
    except Exception as e:
        logging.error(str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
