from typing import Any

from app.auth.adapters.jwt_service import JWTData
from app.auth.router.dependencies import parse_jwt_user_data
from app.utils import AppModel
from fastapi import Depends, Response
from pydantic import Field

from ..service import Service, get_service
from . import router

class GetWtask2Response(AppModel):
    text: str

@router.get("/task2", response_model=GetWtask2Response)
def get_w_task_2(
    text:str
)
