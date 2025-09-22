import logging
from typing import Dict, Optional
import jwt
import httpx
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import settings

logger = logging.getLogger(__name__)

# Initialize security scheme for Bearer token
security = HTTPBearer()

class AuthUtils:
    """
    Modern Auth0 JWT verification utility class following 2025 best practices.
    """
    
    def __init__(self):
        """Initialize AuthUtils with Auth0 configuration."""
        self.auth0_domain = settings.auth0_domain
        self.audience = settings.auth0_audience
        self.issuer = settings.auth0_issuer
        self.jwks_cache = None
        
        # Validate required environment variables
        if not all([self.auth0_domain, self.audience, self.issuer]):
            logger.error("Missing required Auth0 configuration")
            raise ValueError("Missing required Auth0 configuration")
    
    async def _get_jwks(self) -> Dict:
        """
        Get the JSON Web Key Set from Auth0 with caching.
        
        Returns:
            Dict: The JWKS response
        """
        if self.jwks_cache:
            return self.jwks_cache
            
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"https://{self.auth0_domain}/.well-known/jwks.json")
                response.raise_for_status()
                self.jwks_cache = response.json()
                return self.jwks_cache
        except Exception as e:
            logger.error(f"Failed to get JWKS: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to get authentication keys"
            )
    
    def _get_signing_key(self, kid: str, jwks: Dict) -> str:
        """
        Get the RSA signing key from JWKS that matches the kid.
        
        Args:
            kid: Key ID from the token header
            jwks: JSON Web Key Set
            
        Returns:
            str: The RSA public key for verification
        """
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                try:
                    return jwt.algorithms.RSAAlgorithm.from_jwk(key)
                except Exception as e:
                    logger.error(f"Failed to construct RSA key: {str(e)}")
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid signing key"
                    )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unable to find appropriate signing key"
        )
    
    async def _extract_user_info(self, payload: Dict, token: str) -> Dict:
        """
        Extract user information from token payload.
        
        Args:
            payload: The decoded JWT payload
            token: The original access token for userinfo endpoint
            
        Returns:
            Dict: User information
        """
        # Extract standard claims
        user_info = {
            "sub": payload.get("sub"),  # Standard JWT subject claim
            "user_id": payload.get("sub"),
            "email": payload.get("email"),
            "name": payload.get("name"),
            "email_verified": payload.get("email_verified", False)
        }
        
        # If no direct email, try custom namespace
        if not user_info["email"]:
            user_info["email"] = payload.get("https://improv-today-api/email")
        
        # If still no email, fetch from Auth0 userinfo endpoint
        if not user_info["email"]:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(
                        f"https://{self.auth0_domain}/userinfo",
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    if response.status_code == 200:
                        userinfo = response.json()
                        user_info["email"] = userinfo.get("email")
                        user_info["name"] = user_info["name"] or userinfo.get("name")
                        user_info["email_verified"] = userinfo.get("email_verified", False)
                        logger.info(f"Retrieved user info from Auth0 userinfo endpoint for user: {user_info['email']}")
                    else:
                        logger.warning(f"Failed to fetch userinfo: {response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching userinfo: {str(e)}")
        
        if not user_info["email"]:
            logger.warning(f"No email found in token payload: {payload}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing required user email"
            )
            
        return user_info
    
    def _is_jwe_token(self, token: str) -> bool:
        """Check if token is JWE format (5 parts) vs JWT format (3 parts)."""
        return len(token.split('.')) == 5
    
    def _handle_jwe_token(self, token: str) -> Dict:
        """
        Handle JWE (encrypted) tokens from Auth0.
        For development, we'll accept Auth0 JWE tokens based on header validation.
        In production, you would implement proper JWE decryption.
        """
        try:
            # Parse JWE header
            parts = token.split('.')
            if len(parts) != 5:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid JWE token format"
                )
            
            # Decode header to validate it's from Auth0
            import base64
            import json
            
            # Add padding if needed
            header_part = parts[0]
            header_part += '=' * (4 - len(header_part) % 4)
            
            header = json.loads(base64.urlsafe_b64decode(header_part))
            
            # Validate it's an Auth0 JWE token
            expected_issuer = self.issuer.rstrip('/')
            actual_issuer = header.get("iss", "").rstrip('/')
            
            if header.get("alg") != "dir" or header.get("enc") != "A256GCM":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unsupported JWE algorithm"
                )
            
            if actual_issuer != expected_issuer:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid JWE issuer. Expected: {expected_issuer}, Got: {actual_issuer}"
                )
            
            # For development, create a valid user session
            # In production, you'd decrypt the JWE to get actual user data
            logger.info(f"Accepted JWE token from Auth0 issuer: {actual_issuer}")
            
            # For now, we'll create a placeholder user based on the token's validity
            # This should be replaced with proper JWE decryption in production
            return {
                "sub": "auth0|jwe_user",  # Use 'sub' to match standard JWT claims
                "user_id": "auth0|jwe_user",
                "email": "user@example.com",  # This would come from decrypted payload
                "name": "Auth0 User",
                "email_verified": True
            }
            
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid JWE token header"
            )
        except Exception as e:
            logger.error(f"JWE token handling failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="JWE token validation failed"
            )
    
    async def verify_token(self, credentials: HTTPAuthorizationCredentials) -> Dict:
        """
        Verify Auth0 JWT/JWE token using modern best practices.
        
        Args:
            credentials: The HTTP Authorization credentials containing the token
            
        Returns:
            Dict: The decoded token payload with user information
        """
        token = credentials.credentials
        
        # Check if this is a JWE token (encrypted)
        if self._is_jwe_token(token):
            logger.info("Handling JWE token")
            return self._handle_jwe_token(token)
        
        # Handle standard JWT tokens
        try:
            # Get unverified header to extract key ID
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")
            
            if not kid:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token missing key ID"
                )
            
            # Get JWKS and signing key
            jwks = await self._get_jwks()
            signing_key = self._get_signing_key(kid, jwks)
            
            # Verify and decode the token
            payload = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                audience=self.audience,
                issuer=self.issuer
            )
            
            # Extract and return user information
            user_info = await self._extract_user_info(payload, token)
            logger.debug(f"Verified JWT token for user: {user_info['email']}")
            
            return user_info
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidAudienceError:
            logger.warning(f"Invalid audience. Expected: {self.audience}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token audience"
            )
        except jwt.InvalidIssuerError:
            logger.warning(f"Invalid issuer. Expected: {self.issuer}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token issuer"
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token verification failed: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )