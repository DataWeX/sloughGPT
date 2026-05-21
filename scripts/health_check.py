#!/usr/bin/env python3
"""
Health check script for SloughGPT API
Can be used for monitoring or as a simple CLI tool
"""

import sys
import argparse
import requests
import time
from typing import Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class HealthChecker:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 5):
        self.base_url = base_url
        self.timeout = timeout
    
    def check_health(self) -> dict:
        """Basic health check."""
        try:
            r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            return {
                "status": "ok" if r.status_code == 200 else "error",
                "code": r.status_code,
                "data": r.json()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_live(self) -> dict:
        """Kubernetes liveness probe."""
        try:
            r = requests.get(f"{self.base_url}/health/live", timeout=self.timeout)
            return {
                "status": "ok" if r.status_code == 200 else "error",
                "code": r.status_code,
                "data": r.json()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_ready(self) -> dict:
        """Kubernetes readiness probe."""
        try:
            r = requests.get(f"{self.base_url}/health/ready", timeout=self.timeout)
            return {
                "status": "ok" if r.status_code == 200 else "error",
                "code": r.status_code,
                "data": r.json()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_detailed(self) -> dict:
        """Detailed health with system info."""
        try:
            r = requests.get(f"{self.base_url}/health/detailed", timeout=self.timeout)
            return {
                "status": "ok" if r.status_code == 200 else "error",
                "code": r.status_code,
                "data": r.json()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_metrics(self) -> dict:
        """Check metrics endpoint."""
        try:
            r = requests.get(f"{self.base_url}/metrics", timeout=self.timeout)
            return {
                "status": "ok" if r.status_code == 200 else "error",
                "code": r.status_code,
                "data": r.json()
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_all(self) -> dict:
        """Run all checks."""
        return {
            "health": self.check_health(),
            "live": self.check_live(),
            "ready": self.check_ready(),
            "detailed": self.check_detailed(),
            "metrics": self.check_metrics(),
        }
    
    def print_results(self, results: dict):
        """Print formatted results."""
        print("=" * 60)
        print("SloughGPT Health Check")
        print("=" * 60)
        print(f"Base URL: {self.base_url}")
        print()
        
        for name, result in results.items():
            status = result.get("status", "unknown")
            status_symbol = "✓" if status == "ok" else "✗"
            print(f"{status_symbol} {name.upper():12} [{status:6}]")
            
            if status == "ok":
                data = result.get("data", {})
                for key, value in data.items():
                    if key not in ["system"]:
                        print(f"    {key}: {value}")
            else:
                print(f"    Error: {result.get('error', 'Unknown')}")
            
            print()


def main():
    parser = argparse.ArgumentParser(description="SloughGPT Health Check")
    parser.add_argument("--url", "-u", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--timeout", "-t", type=int, default=5, help="Request timeout")
    parser.add_argument("--watch", "-w", action="store_true", help="Watch mode")
    parser.add_argument("--interval", type=int, default=5, help="Watch interval (seconds)")
    args = parser.parse_args()
    
    checker = HealthChecker(args.url, args.timeout)
    
    if args.watch:
        print(f"Watching health status (Ctrl+C to stop)...\n")
        try:
            while True:
                results = checker.check_all()
                checker.print_results(results)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        results = checker.check_all()
        checker.print_results(results)
        
        # Exit code based on status
        all_ok = all(r.get("status") == "ok" for r in results.values())
        sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
