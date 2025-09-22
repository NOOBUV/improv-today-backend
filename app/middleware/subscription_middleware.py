"""Middleware for handling subscription-related requests and responses."""
import logging
from typing import Dict, Any
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class SubscriptionMiddleware(BaseHTTPMiddleware):
    """
    Middleware to handle subscription-related HTTP responses and provide
    graceful degradation for expired subscriptions.
    """
    
    SUBSCRIPTION_REQUIRED_PATHS = [
        "/api/ava/conversation",
        "/api/conversation",
        "/api/sessions/create",
        # Add more paths that require subscriptions
    ]
    
    async def dispatch(self, request: Request, call_next):
        """
        Process the request and handle subscription-related responses.
        
        Args:
            request: The incoming HTTP request
            call_next: The next middleware or endpoint handler
            
        Returns:
            Response: The HTTP response, potentially modified for subscription handling
        """
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as exc:
            if exc.status_code == status.HTTP_402_PAYMENT_REQUIRED:
                # Handle subscription required responses with enhanced error details
                return self._create_subscription_required_response(request, exc)
            raise exc
        except Exception as e:
            logger.error(f"Unexpected error in subscription middleware: {str(e)}")
            raise
    
    def _create_subscription_required_response(
        self, 
        request: Request, 
        exc: HTTPException
    ) -> JSONResponse:
        """
        Create a standardized subscription required response.
        
        Args:
            request: The HTTP request
            exc: The HTTPException that triggered this response
            
        Returns:
            JSONResponse: Standardized subscription required response
        """
        response_data = {
            "error": "subscription_required",
            "message": exc.detail,
            "status_code": 402,
            "path": str(request.url.path),
            "subscription_info": {
                "upgrade_url": "/api/subscriptions/plans",
                "checkout_url": "/api/subscriptions/checkout",
                "status_url": "/api/subscriptions/status"
            },
            "timestamp": "2024-09-12T03:45:00Z"  # Could be made dynamic
        }
        
        # Add trial information if available from headers
        headers = exc.headers or {}
        if "X-Trial-Expired" in headers:
            response_data["trial_info"] = {
                "trial_expired": True,
                "message": "Your free trial has ended. Subscribe to continue using premium features."
            }
        
        logger.info(f"Subscription required for path {request.url.path}: {exc.detail}")
        
        return JSONResponse(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            content=response_data,
            headers={
                "X-Subscription-Required": "true",
                "Access-Control-Allow-Origin": "*",  # Adjust for your CORS policy
                "Content-Type": "application/json"
            }
        )
    
    def _is_subscription_required_path(self, path: str) -> bool:
        """
        Check if the given path requires a subscription.
        
        Args:
            path: The request path
            
        Returns:
            bool: True if subscription is required for this path
        """
        return any(path.startswith(required_path) for required_path in self.SUBSCRIPTION_REQUIRED_PATHS)


async def handle_subscription_error(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Global handler for subscription-related errors.
    
    Args:
        request: The HTTP request
        exc: The subscription-related HTTP exception
        
    Returns:
        JSONResponse: Formatted error response
    """
    if exc.status_code == status.HTTP_402_PAYMENT_REQUIRED:
        return JSONResponse(
            status_code=402,
            content={
                "error": "subscription_required",
                "message": exc.detail,
                "path": str(request.url.path),
                "subscription_required": True,
                "action": "Please upgrade your subscription to access this feature."
            },
            headers={"X-Subscription-Required": "true"}
        )
    
    # Default error handling
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "request_failed", "message": exc.detail}
    )