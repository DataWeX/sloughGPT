import socket
import ipaddress
import logging
from urllib.parse import urlparse
from schemas.common import raise_error

logger = logging.getLogger("slo.infra.ssrf")


def is_private_ip(hostname: str) -> bool:
    """Check if hostname resolves to a private/loopback IP (SSRF protection)."""
    try:
        addrinfos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for family, _, _, _, sockaddr in addrinfos:
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return True
    except (socket.gaierror, ValueError):
        pass
    return False


def validate_url_not_private(url: str) -> None:
    """Validate that a URL does not point to a private/internal IP (SSRF protection)."""
    try:
        parsed = urlparse(url)
        if parsed.hostname and is_private_ip(parsed.hostname):
            raise_error(f"URL points to a private/internal address: {url}", "E_BAD_REQUEST", status_code=400)
    except Exception as e:
        if "E_BAD_REQUEST" in str(e):
            raise
        logger.warning("URL validation failed: %s", e)
