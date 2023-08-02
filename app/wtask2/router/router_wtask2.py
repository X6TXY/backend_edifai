from typing import List

from app.auth.adapters.jwt_service import JWTData
from app.auth.router.dependencies import parse_jwt_user_data
from app.utils import AppModel
from fastapi import Depends, HTTPException, Response, status
from pydantic import Field, ValidationError
from pymongo.cursor import Cursor

from ..repository.repository import Wtask2Repository
from ..service import Service, get_service
from . import router


class GetAnswerRequest(AppModel):
    request: str


class ResponseData(AppModel):
    date: str
    response: str
    request: str
    score: str


class GetAnswerResponse(AppModel):
    response: str
    score: str


class GetScoreResponse(AppModel):
    response: str


class GetScoreRequest(AppModel):
    request: str


class GetDatesResponse(AppModel):
    date: str
    count: int


@router.post("/get_answer", response_model=GetAnswerResponse)
def get_answer(
    request: GetAnswerRequest,
    jwt_data: JWTData = Depends(parse_jwt_user_data),
    svc: Service = Depends(get_service),
) -> dict[str, str]:
    user = svc.repository.get_user_by_id(jwt_data.user_id)
    try:
        if not svc.repository.vector_store:
            svc.repository.load_vector_store()
        response = svc.repository.get_answer(request.request)
        score = svc.repository.get_score(request.request)
        return GetAnswerResponse(response=response, id=user["_id"], score=score)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/get_dates", response_model=dict[str, int])
def get_dates(
    jwt_data: JWTData = Depends(parse_jwt_user_data),
    svc: Service = Depends(get_service),
) -> dict[str, int]:
    user = svc.repository.get_user_by_id(jwt_data.user_id)
    try:
        dates = svc.repository.get_dates()
        return dates
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get("/get_responses", response_model=List[ResponseData])
def get_responses_by_user_id(
    jwt_data: JWTData = Depends(parse_jwt_user_data),
    svc: Service = Depends(get_service),
) -> List[ResponseData]:
    user = svc.repository.get_user_by_id(jwt_data.user_id)
    try:
        responses_cursor = svc.repository.get_responses_by_user_id(str(user["_id"]))
        responses = [
            ResponseData(
                date=response["date"],
                response=response["response"],
                request=response["request"],
                score=response["score"],
            )
            for response in responses_cursor
        ]
        return responses
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
