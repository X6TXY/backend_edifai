from app.utils import AppModel
from fastapi import Depends, HTTPException, status

from ..service import Service, get_service
from . import router


class RegisterUserRequest(AppModel):
    email: str
    password: str


class RegisterUserResponse(AppModel):
    email: str


@router.post(
    "/users", status_code=status.HTTP_200_OK, response_model=RegisterUserResponse
)
def register_user(
    input: RegisterUserRequest = Depends(),
    svc: Service = Depends(get_service),
) -> dict[str, str]:
    if svc.repository.get_user_by_email(input.email):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Email is already taken.",
        )

    svc.repository.create_user(input.dict())

    return RegisterUserResponse(email=input.email)
