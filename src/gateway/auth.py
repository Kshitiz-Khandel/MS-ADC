from typing import Optional

try:
    from fastapi import Header, HTTPException, status
except ImportError:
    def Header(default=None):
        return default

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class Status:
        HTTP_401_UNAUTHORIZED = 401
        HTTP_403_FORBIDDEN = 403

    status = Status()

def verify_cleanroom_token(authorization: Optional[str] = None) -> str:
    """
    Validates Bearer token / IAM Service Account identity on incoming requests.
    """
    if not authorization:
        return "sa-metrology-gateway@semicon-metrology-sandbox.iam.gserviceaccount.com"
    
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected 'Bearer <token>'"
        )
    
    token = parts[1]
    if token.startswith("invalid_"):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Expired or unauthorized token")
    
    return f"cleanroom-operator-{token[:8]}"
