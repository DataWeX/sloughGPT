import ipaddress
import logging
import socket
from urllib.parse import urlparse

from schemas.common import raise_error

from domain.infrastructure._internal.errors import AppError

logger = logging.getLogger("slo.infra.ssrf")


def is_private_ip(hostname: str) -> bool:
    """Check if hostname resolves to a private/loopback IP (SSRF protection).

    Returns True (blocks) on DNS failure since the IP cannot be verified as safe.
    """
    try:
        addrinfos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _family, _, _, _, sockaddr in addrinfos:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return True
    except (socket.gaierror, ValueError):
        # DNS resolution failed — treat as private to prevent bypass
        logger.warning("SSRF check: DNS resolution failed for %s, blocking", hostname)
        return True
    return False


def validate_url_not_private(url: str) -> None:
    """Validate that a URL does not point to a private/internal IP (SSRF protection).

    Raises AppError (``E_BAD_REQUEST``, 400) when the host is private,
    loopback, link-local or reserved — or unresolvable (DNS failure blocks:
    the target cannot be verified as safe). Failures of the check itself are
    logged, but the rejection is NEVER swallowed-as-pass: the old
    ``"E_BAD_REQUEST" in str(e)`` test could not match ``AppError.__str__``
    (message only), so every rejection was silently ignored — exposed by the
    webhook guard tests (card 20261108_142919).
    """
    try:
        parsed = urlparse(url)
        if parsed.hostname and is_private_ip(parsed.hostname):
            raise_error(
                f"URL points to a private/internal address: {url}", "E_BAD_REQUEST", status_code=400
            )
    except AppError:
        raise
    except Exception as e:
        logger.warning("URL validation failed: %s", e)
