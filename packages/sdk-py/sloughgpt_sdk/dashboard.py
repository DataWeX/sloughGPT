"""
SloughGPT SDK - Dashboard
Usage analytics and reporting dashboard.
"""

import time
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class DashboardMetrics:
    """Dashboard metrics summary."""
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0
    active_keys: int = 0
    active_users: int = 0
    avg_response_time_ms: float = 0
    success_rate: float = 0
    period_start: float = 0
    period_end: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "active_keys": self.active_keys,
            "active_users": self.active_users,
            "avg_response_time_ms": self.avg_response_time_ms,
            "success_rate": self.success_rate,
            "period_start": self.period_start,
            "period_end": self.period_end,
        }


@dataclass
class TimeSeriesPoint:
    """A single point in a time series."""
    timestamp: float
    value: float
    label: Optional[str] = None


@dataclass
class UsageBreakdown:
    """Breakdown of usage by category."""
    category: str
    count: int
    percentage: float
    cost: float


class UsageDashboard:
    """
    Usage analytics and reporting dashboard.
    
    Example:
    
    ```python
    from sloughgpt_sdk.dashboard import UsageDashboard
    
    dashboard = UsageDashboard()
    
    # Record usage
    dashboard.record_request(
        key_id="sk_xxx",
        customer_id="cus_xxx",
        tokens=100,
        latency_ms=50,
        cached=False
    )
    
    # Get metrics
    metrics = dashboard.get_metrics(period="7d")
    
    # Get charts data
    requests_chart = dashboard.get_requests_timeseries(period="7d")
    cost_chart = dashboard.get_cost_timeseries(period="30d")
    
    # Get breakdown
    breakdown = dashboard.get_usage_breakdown()
    
    # Export report
    report = dashboard.generate_report(format="json")
    ```
    """
    
    def __init__(self, storage_path: str = "./.usage_data.json"):
        """
        Initialize usage dashboard.
        
        Args:
            storage_path: Path to store usage data.
        """
        self._storage_path = storage_path
        self._requests: List[Dict[str, Any]] = []
        self._daily_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "requests": 0, "tokens": 0, "cost": 0, "errors": 0
        })
        self._key_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "requests": 0, "tokens": 0, "cost": 0, "errors": 0, "last_seen": 0
        })
        self._customer_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "requests": 0, "tokens": 0, "cost": 0, "keys": set(), "errors": 0
        })
        self._load_data()
    
    def _load_data(self):
        """Load usage data from storage."""
        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)
                self._requests = data.get("requests", [])
                self._daily_stats = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0, "errors": 0}, data.get("daily_stats", {}))
                self._key_stats = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0, "errors": 0, "last_seen": 0}, data.get("key_stats", {}))
                self._customer_stats = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0, "keys": set(), "errors": 0}, data.get("customer_stats", {}))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    def _save_data(self):
        """Save usage data to storage."""
        def convert_stats(stats_dict):
            result = {}
            for k, v in stats_dict.items():
                if isinstance(v.get("keys"), set):
                    v = {**v, "keys": list(v["keys"])}
                result[k] = v
            return result
        
        data = {
            "requests": self._requests[-10000:],
            "daily_stats": dict(self._daily_stats),
            "key_stats": convert_stats(self._key_stats),
            "customer_stats": convert_stats(self._customer_stats),
        }
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def record_request(
        self,
        key_id: str,
        customer_id: str,
        tokens: int = 0,
        latency_ms: float = 0,
        cached: bool = False,
        success: bool = True,
        error: Optional[str] = None,
        endpoint: str = "/generate",
        model: str = "gpt2",
    ):
        """Record an API request."""
        now = time.time()
        date_key = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
        
        cost = self._calculate_cost(tokens)
        
        request = {
            "timestamp": now,
            "key_id": key_id,
            "customer_id": customer_id,
            "tokens": tokens,
            "latency_ms": latency_ms,
            "cached": cached,
            "success": success,
            "error": error,
            "endpoint": endpoint,
            "model": model,
            "cost": cost,
        }
        self._requests.append(request)
        
        self._daily_stats[date_key]["requests"] += 1
        self._daily_stats[date_key]["tokens"] += tokens
        self._daily_stats[date_key]["cost"] += cost
        if not success:
            self._daily_stats[date_key]["errors"] += 1
        
        self._key_stats[key_id]["requests"] += 1
        self._key_stats[key_id]["tokens"] += tokens
        self._key_stats[key_id]["cost"] += cost
        self._key_stats[key_id]["last_seen"] = now
        if not success:
            self._key_stats[key_id]["errors"] += 1
        
        self._customer_stats[customer_id]["requests"] += 1
        self._customer_stats[customer_id]["tokens"] += tokens
        self._customer_stats[customer_id]["cost"] += cost
        self._customer_stats[customer_id]["keys"].add(key_id)
        if not success:
            self._customer_stats[customer_id]["errors"] += 1
        
        self._save_data()
    
    @staticmethod
    def _calculate_cost(tokens: int) -> float:
        """Calculate cost for tokens."""
        return tokens * 0.00001
    
    def get_metrics(
        self,
        period: str = "7d",
        customer_id: Optional[str] = None,
        key_id: Optional[str] = None,
    ) -> DashboardMetrics:
        """
        Get dashboard metrics for a period.
        
        Args:
            period: Time period (1d, 7d, 30d, 90d)
            customer_id: Optional filter by customer
            key_id: Optional filter by key
        
        Returns:
            DashboardMetrics object.
        """
        period_days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}.get(period, 7)
        period_start = time.time() - (period_days * 86400)
        
        requests = self._requests
        if customer_id:
            requests = [r for r in requests if r["customer_id"] == customer_id]
        if key_id:
            requests = [r for r in requests if r["key_id"] == key_id]
        
        requests = [r for r in requests if r["timestamp"] >= period_start]
        
        total_requests = len(requests)
        total_tokens = sum(r["tokens"] for r in requests)
        total_cost = sum(r["cost"] for r in requests)
        total_errors = sum(1 for r in requests if not r["success"])
        total_latency = sum(r["latency_ms"] for r in requests)
        
        active_keys = len(set(r["key_id"] for r in requests))
        active_users = len(set(r["customer_id"] for r in requests))
        
        return DashboardMetrics(
            total_requests=total_requests,
            total_tokens=total_tokens,
            total_cost=total_cost,
            active_keys=active_keys,
            active_users=active_users,
            avg_response_time_ms=total_latency / total_requests if total_requests else 0,
            success_rate=(total_requests - total_errors) / total_requests if total_requests else 1,
            period_start=period_start,
            period_end=time.time(),
        )
    
    def get_requests_timeseries(
        self,
        period: str = "7d",
        granularity: str = "day",
        customer_id: Optional[str] = None,
    ) -> List[TimeSeriesPoint]:
        """Get requests over time."""
        period_days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}.get(period, 7)
        period_start = time.time() - (period_days * 86400)
        
        requests = [r for r in self._requests if r["timestamp"] >= period_start]
        if customer_id:
            requests = [r for r in requests if r["customer_id"] == customer_id]
        
        if granularity == "hour":
            buckets = defaultdict(int)
            for r in requests:
                dt = datetime.fromtimestamp(r["timestamp"])
                key = dt.strftime("%Y-%m-%d %H:00")
                buckets[key] += 1
            
            return [
                TimeSeriesPoint(
                    timestamp=datetime.strptime(k, "%Y-%m-%d %H:00").timestamp(),
                    value=v,
                    label=k,
                )
                for k, v in sorted(buckets.items())
            ]
        else:
            buckets = defaultdict(int)
            for r in requests:
                dt = datetime.fromtimestamp(r["timestamp"])
                key = dt.strftime("%Y-%m-%d")
                buckets[key] += 1
            
            return [
                TimeSeriesPoint(
                    timestamp=datetime.strptime(k, "%Y-%m-%d").timestamp(),
                    value=v,
                    label=k,
                )
                for k, v in sorted(buckets.items())
            ]
    
    def get_cost_timeseries(
        self,
        period: str = "30d",
        granularity: str = "day",
    ) -> List[TimeSeriesPoint]:
        """Get cost over time."""
        period_days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}.get(period, 30)
        period_start = time.time() - (period_days * 86400)
        
        requests = [r for r in self._requests if r["timestamp"] >= period_start]
        
        if granularity == "hour":
            buckets = defaultdict(float)
            for r in requests:
                dt = datetime.fromtimestamp(r["timestamp"])
                key = dt.strftime("%Y-%m-%d %H:00")
                buckets[key] += r["cost"]
            
            return [
                TimeSeriesPoint(
                    timestamp=datetime.strptime(k, "%Y-%m-%d %H:00").timestamp(),
                    value=round(v, 4),
                    label=k,
                )
                for k, v in sorted(buckets.items())
            ]
        else:
            buckets = defaultdict(float)
            for r in requests:
                dt = datetime.fromtimestamp(r["timestamp"])
                key = dt.strftime("%Y-%m-%d")
                buckets[key] += r["cost"]
            
            return [
                TimeSeriesPoint(
                    timestamp=datetime.strptime(k, "%Y-%m-%d").timestamp(),
                    value=round(v, 4),
                    label=k,
                )
                for k, v in sorted(buckets.items())
            ]
    
    def get_usage_breakdown(
        self,
        period: str = "7d",
        group_by: str = "endpoint",
    ) -> List[UsageBreakdown]:
        """Get usage breakdown by category."""
        period_days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}.get(period, 7)
        period_start = time.time() - (period_days * 86400)
        
        requests = [r for r in self._requests if r["timestamp"] >= period_start]
        
        buckets = defaultdict(lambda: {"count": 0, "cost": 0})
        for r in requests:
            if group_by == "endpoint":
                key = r["endpoint"]
            elif group_by == "model":
                key = r["model"]
            elif group_by == "key":
                key = r["key_id"]
            else:
                key = r["customer_id"]
            buckets[key]["count"] += 1
            buckets[key]["cost"] += r["cost"]
        
        total = sum(b["count"] for b in buckets.values())
        
        return [
            UsageBreakdown(
                category=k,
                count=v["count"],
                percentage=(v["count"] / total * 100) if total else 0,
                cost=v["cost"],
            )
            for k, v in sorted(buckets.items(), key=lambda x: x[1]["count"], reverse=True)
        ]
    
    def get_top_customers(
        self,
        period: str = "30d",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get top customers by usage."""
        period_days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}.get(period, 30)
        period_start = time.time() - (period_days * 86400)
        
        requests = [r for r in self._requests if r["timestamp"] >= period_start]
        
        customer_usage = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0})
        for r in requests:
            customer_usage[r["customer_id"]]["requests"] += 1
            customer_usage[r["customer_id"]]["tokens"] += r["tokens"]
            customer_usage[r["customer_id"]]["cost"] += r["cost"]
        
        sorted_customers = sorted(
            customer_usage.items(),
            key=lambda x: x[1]["requests"],
            reverse=True
        )[:limit]
        
        return [
            {
                "customer_id": cid,
                "requests": stats["requests"],
                "tokens": stats["tokens"],
                "cost": stats["cost"],
            }
            for cid, stats in sorted_customers
        ]
    
    def get_top_keys(
        self,
        period: str = "30d",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get top API keys by usage."""
        period_days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}.get(period, 30)
        period_start = time.time() - (period_days * 86400)
        
        requests = [r for r in self._requests if r["timestamp"] >= period_start]
        
        key_usage = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0, "errors": 0})
        for r in requests:
            key_usage[r["key_id"]]["requests"] += 1
            key_usage[r["key_id"]]["tokens"] += r["tokens"]
            key_usage[r["key_id"]]["cost"] += r["cost"]
            if not r["success"]:
                key_usage[r["key_id"]]["errors"] += 1
        
        sorted_keys = sorted(
            key_usage.items(),
            key=lambda x: x[1]["requests"],
            reverse=True
        )[:limit]
        
        return [
            {
                "key_id": kid,
                "requests": stats["requests"],
                "tokens": stats["tokens"],
                "cost": stats["cost"],
                "errors": stats["errors"],
                "error_rate": stats["errors"] / stats["requests"] if stats["requests"] else 0,
            }
            for kid, stats in sorted_keys
        ]
    
    def generate_report(
        self,
        period: str = "30d",
        format: str = "json",
        customer_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a usage report.
        
        Args:
            period: Time period.
            format: Output format (json, summary).
            customer_id: Optional customer filter.
        
        Returns:
            Report dictionary.
        """
        metrics = self.get_metrics(period, customer_id)
        requests_ts = self.get_requests_timeseries(period)
        cost_ts = self.get_cost_timeseries(period)
        breakdown = self.get_usage_breakdown(period)
        
        report = {
            "period": period,
            "generated_at": time.time(),
            "metrics": metrics.to_dict(),
            "requests_timeseries": [
                {"timestamp": p.timestamp, "value": p.value, "label": p.label}
                for p in requests_ts
            ],
            "cost_timeseries": [
                {"timestamp": p.timestamp, "value": p.value, "label": p.label}
                for p in cost_ts
            ],
            "breakdown": [
                {"category": b.category, "count": b.count, "percentage": b.percentage, "cost": b.cost}
                for b in breakdown
            ],
            "top_customers": self.get_top_customers(period),
            "top_keys": self.get_top_keys(period),
        }
        
        return report
    
    def export_csv(self, period: str = "30d", customer_id: Optional[str] = None) -> str:
        """Export usage data as CSV."""
        period_days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}.get(period, 30)
        period_start = time.time() - (period_days * 86400)
        
        requests = [r for r in self._requests if r["timestamp"] >= period_start]
        if customer_id:
            requests = [r for r in requests if r["customer_id"] == customer_id]
        
        lines = ["timestamp,customer_id,key_id,endpoint,model,tokens,latency_ms,cached,success,cost"]
        for r in requests:
            lines.append(
                f"{r['timestamp']},{r['customer_id']},{r['key_id']},{r['endpoint']},"
                f"{r['model']},{r['tokens']},{r['latency_ms']},{r['cached']},"
                f"{r['success']},{r['cost']}"
            )
        
        return "\n".join(lines)
