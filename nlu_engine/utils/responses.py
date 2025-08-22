from typing import TypedDict


class BaseResponse(TypedDict, total=False):
    """
    Standardized response schema for the NLU Engine.

    This type ensures that all responses returned by the engine
    follow a consistent structure, whether success or error.

    Attributes:
        status (str): Either 'success' or 'error'.
        code (str): A short response code (e.g., "DZ_SUCCESS_200", "DZ_ERROR_400").
        message (str): Human-readable explanation of the response.
        data (dict | str | None, optional): The result payload (if any).
        errors (dict | str | None, optional): Error details (if applicable).
    """

    status: str
    code: str
    message: str
    data: dict | str | None
    errors: dict | str | None


def success_response(
    code: str = "DZ_SUCCESS_200", message: str = "Request completed successfully", data: dict | str | None = None
) -> BaseResponse:
    """
    Generate a standardized success response.

    Args:
        code (str): Status code representing the result. Defaults to DZ_200.
        message (str): Human-readable message describing the success. Defaults to "Operation successful".
        data (Optional[Union[dict, list]]): Payload of the operation, typically a dict or list. Defaults to None.

    Returns:
        dict[str, str | dict | None]: Standardized success response.
    """
    return {"status": "success", "code": code, "message": message, "data": data, "errors": None}


def error_response(
    code: str = "DZ_ERROR_400", message: str = "An error occurred", errors: dict | str | None = None
) -> BaseResponse:
    """
    Generate a standardized error response.

    Args:
        code (str): Status code representing the error. Defaults to DZ_400.
        message (str): Human-readable message describing the error. Defaults to "An error occurred".
        errors (Optional[Union[dict, list, str]]): Optional detailed error info. Defaults to None.

    Returns:
        dict[str, str | dict | None]: Standardized error response.
    """
    return {"status": "error", "code": code, "message": message, "data": None, "errors": errors}
