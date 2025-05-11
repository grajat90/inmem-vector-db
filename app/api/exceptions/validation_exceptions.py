from fastapi import status

from app.api.exceptions.base import AppException


class InvalidHyperparameterException(AppException):
    def __init__(self, hyperparameter: str, value: str | int):
        super().__init__(
            message=f"Invalid hyperparameter: {hyperparameter} with value: {value}",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"hyperparameter": hyperparameter, "value": value},
        )
