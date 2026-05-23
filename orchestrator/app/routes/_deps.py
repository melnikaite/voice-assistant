"""Shared dependencies for HTTP route modules.

Lives here (instead of main.py) so each routes/<feature>.py can import
the cookie-auth dependency without pulling main.py and causing import
cycles.
"""

from fastapi import Cookie, HTTPException

from ..storage import get_session as auth_get_session


# Browser session cookie name.  HttpOnly + SameSite=Lax so a malicious
# tab can't read it via JS but an in-tab POST to /api/users/... still
# carries it.  Insecure (no `Secure` flag) because we're served over
# plain HTTP on localhost; flip when fronted by HTTPS.
SESSION_COOKIE = "va_session"


async def _current_user(
    va_session: str | None = Cookie(default=None),
) -> dict:
    """FastAPI dependency: resolve the cookie session → profile.

    Raises 401 if the cookie is missing, invalid, or expired.  Endpoints
    that want auth declare `user: dict = Depends(_current_user)`; ones
    that don't (login, setup) skip it.
    """
    if not va_session:
        raise HTTPException(401, "not authenticated")
    sess = await auth_get_session(va_session)
    if not sess:
        raise HTTPException(401, "session expired or invalid")
    return sess
