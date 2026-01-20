"""
JWT Authentication Module for NBA Props API

Optional JWT authentication for protecting sensitive endpoints.
Configure via environment variables:
    - JWT_SECRET_KEY: Secret key for signing tokens
    - JWT_ALGORITHM: Algorithm (default: HS256)
    - JWT_ACCESS_TOKEN_EXPIRE_MINUTES: Token expiration (default: 30)
    - API_KEY: Simple API key for basic auth (optional alternative to JWT)

Usage:
    from backend.auth import get_current_user, create_access_token

    @app.get("/api/predictions/{date}")
    async def get_predictions(
        date: str,
        current_user: dict = Depends(get_current_user)
    ):
        # Protected endpoint
        pass
"""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext


# ============== CONFIGURATION ==============

# JWT Configuration
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "your-secret-key-change-in-production")
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Simple API Key (alternative to JWT)
API_KEY = os.environ.get("API_KEY", None)

# Enable/disable authentication
AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "false").lower() == "true"

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Bearer token security
security = HTTPBearer(auto_error=False)


# ============== PASSWORD UTILITIES ==============

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


# ============== JWT TOKEN UTILITIES ==============

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.

    Args:
        data: Payload data to encode in token
        expires_delta: Token expiration time (default: 30 minutes)

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode and verify a JWT token.

    Args:
        token: JWT token string

    Returns:
        Decoded token payload

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ============== AUTHENTICATION DEPENDENCIES ==============

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[Dict[str, Any]]:
    """Get current authenticated user from JWT token or API key.

    This dependency can be used to protect endpoints.

    Args:
        credentials: Bearer token from Authorization header
        api_key: API key from X-API-Key header

    Returns:
        User info dict or None if auth is disabled

    Raises:
        HTTPException: If authentication fails
    """
    # If auth is disabled, allow all requests
    if not AUTH_ENABLED:
        return {"user_id": "anonymous", "username": "anonymous"}

    # Try API key authentication first
    if api_key and API_KEY:
        if api_key == API_KEY:
            return {"user_id": "api_key_user", "username": "api_key_user"}

    # Try JWT authentication
    if credentials:
        token = credentials.credentials
        try:
            payload = decode_access_token(token)
            user_id = payload.get("sub")
            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return {
                "user_id": user_id,
                "username": payload.get("username"),
                "email": payload.get("email"),
            }
        except HTTPException:
            raise

    # No valid authentication provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. Provide Bearer token or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> Optional[Dict[str, Any]]:
    """Get current user but don't require authentication.

    Useful for endpoints that provide different features for authenticated users.

    Args:
        credentials: Bearer token from Authorization header
        api_key: API key from X-API-Key header

    Returns:
        User info dict or None if not authenticated
    """
    try:
        return await get_current_user(credentials, api_key)
    except HTTPException:
        return None


# ============== AUTHENTICATION ENDPOINTS ==============

def add_auth_endpoints(app):
    """Add authentication endpoints to FastAPI app.

    Adds:
        - POST /api/auth/token - Generate JWT token
        - GET /api/auth/verify - Verify current token

    Args:
        app: FastAPI application instance
    """
    from pydantic import BaseModel

    class TokenRequest(BaseModel):
        username: str
        password: str

    class TokenResponse(BaseModel):
        access_token: str
        token_type: str
        expires_in: int

    class VerifyResponse(BaseModel):
        valid: bool
        user_id: Optional[str] = None
        username: Optional[str] = None
        expires_at: Optional[str] = None

    @app.post("/api/auth/token", response_model=TokenResponse)
    async def login(request: TokenRequest):
        """Generate JWT access token.

        For demo purposes, accepts any username/password.
        In production, verify against a user database.
        """
        # TODO: Replace with real user verification
        # For now, accept any credentials for demo
        user_data = {
            "sub": request.username,
            "username": request.username,
        }

        access_token = create_access_token(data=user_data)

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        )

    @app.get("/api/auth/verify", response_model=VerifyResponse)
    async def verify_token(current_user: Dict[str, Any] = Depends(get_current_user)):
        """Verify current JWT token is valid."""
        return VerifyResponse(
            valid=True,
            user_id=current_user.get("user_id"),
            username=current_user.get("username"),
        )


# ============== RATE LIMITING (OPTIONAL) ==============

class RateLimiter:
    """Simple in-memory rate limiter.

    For production, use Redis or a dedicated rate limiting service.
    """

    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list] = {}

    def check_rate_limit(self, identifier: str) -> bool:
        """Check if identifier has exceeded rate limit.

        Args:
            identifier: User ID, IP address, or API key

        Returns:
            True if under limit, False if exceeded
        """
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.window_seconds)

        # Initialize or clean old requests
        if identifier not in self._requests:
            self._requests[identifier] = []

        # Remove old requests outside window
        self._requests[identifier] = [
            ts for ts in self._requests[identifier]
            if ts > cutoff
        ]

        # Check limit
        if len(self._requests[identifier]) >= self.max_requests:
            return False

        # Add current request
        self._requests[identifier].append(now)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=100, window_seconds=3600)


async def check_rate_limit(
    current_user: Optional[Dict[str, Any]] = Depends(get_optional_user)
) -> None:
    """Rate limiting dependency.

    Args:
        current_user: Current authenticated user

    Raises:
        HTTPException: If rate limit exceeded
    """
    if not AUTH_ENABLED:
        return

    # Use user_id if authenticated, otherwise skip rate limiting
    if current_user:
        identifier = current_user.get("user_id", "anonymous")
    else:
        identifier = "anonymous"

    if not rate_limiter.check_rate_limit(identifier):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
        )
