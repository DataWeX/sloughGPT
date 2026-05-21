"""
SloughGPT SDK - Billing
Subscription and payment management.
"""

import hashlib
import hmac
import secrets
import time
import json
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta


class BillingCycle(Enum):
    """Billing cycle options."""
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class SubscriptionStatus(Enum):
    """Subscription status."""
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"
    TRIALING = "trialing"


class PaymentStatus(Enum):
    """Payment status."""
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REFUNDED = "refunded"
    DISPUTED = "disputed"


@dataclass
class Plan:
    """Subscription plan definition."""
    id: str
    name: str
    description: str
    price_monthly: float
    price_yearly: float
    features: List[str]
    limits: Dict[str, Any]
    is_active: bool = True
    
    def get_price(self, cycle: BillingCycle) -> float:
        """Get price for billing cycle."""
        if cycle == BillingCycle.MONTHLY:
            return self.price_monthly
        elif cycle == BillingCycle.QUARTERLY:
            return self.price_monthly * 3 * 0.95
        elif cycle == BillingCycle.YEARLY:
            return self.price_yearly * 0.8
        return self.price_monthly
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "price_monthly": self.price_monthly,
            "price_yearly": self.price_yearly,
            "features": self.features,
            "limits": self.limits,
            "is_active": self.is_active,
        }


@dataclass
class Subscription:
    """Customer subscription."""
    id: str
    customer_id: str
    plan_id: str
    status: SubscriptionStatus
    billing_cycle: BillingCycle
    current_period_start: float
    current_period_end: float
    created_at: float = field(default_factory=time.time)
    cancelled_at: Optional[float] = None
    trial_end: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Check if subscription is currently active."""
        return self.status == SubscriptionStatus.ACTIVE
    
    def is_trial(self) -> bool:
        """Check if in trial period."""
        if self.trial_end:
            return time.time() < self.trial_end
        return False
    
    def get_remaining_days(self) -> int:
        """Get remaining days in current period."""
        remaining = self.current_period_end - time.time()
        return max(0, int(remaining / 86400))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "plan_id": self.plan_id,
            "status": self.status.value,
            "billing_cycle": self.billing_cycle.value,
            "current_period_start": self.current_period_start,
            "current_period_end": self.current_period_end,
            "created_at": self.created_at,
            "cancelled_at": self.cancelled_at,
            "trial_end": self.trial_end,
            "metadata": self.metadata,
        }


@dataclass
class Invoice:
    """Billing invoice."""
    id: str
    customer_id: str
    subscription_id: str
    amount: float
    currency: str = "usd"
    status: PaymentStatus = PaymentStatus.PENDING
    period_start: float = 0
    period_end: float = 0
    created_at: float = field(default_factory=time.time)
    paid_at: Optional[float] = None
    items: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_paid(self) -> bool:
        """Check if invoice is paid."""
        return self.status == PaymentStatus.SUCCEEDED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "subscription_id": self.subscription_id,
            "amount": self.amount,
            "currency": self.currency,
            "status": self.status.value,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "created_at": self.created_at,
            "paid_at": self.paid_at,
            "items": self.items,
            "metadata": self.metadata,
        }


@dataclass
class UsageRecord:
    """Record of API usage."""
    id: str
    customer_id: str
    key_id: str
    requests: int
    tokens_used: int
    timestamp: float = field(default_factory=time.time)
    cost: float = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "key_id": self.key_id,
            "requests": self.requests,
            "tokens_used": self.tokens_used,
            "timestamp": self.timestamp,
            "cost": self.cost,
        }


class BillingManager:
    """
    Manages subscriptions, billing, and payments.
    
    Example:
    
    ```python
    from sloughgpt_sdk.billing import BillingManager, Plan, BillingCycle
    
    billing = BillingManager()
    
    # Define plans
    billing.create_plan(
        id="pro",
        name="Pro Plan",
        price_monthly=29.99,
        price_yearly=299.99,
        features=["Unlimited requests", "Priority support"],
        limits={"requests_per_day": 10000}
    )
    
    # Create subscription
    subscription = billing.create_subscription(
        customer_id="cus_123",
        plan_id="pro",
        billing_cycle=BillingCycle.MONTHLY
    )
    
    # Generate invoice
    invoice = billing.create_invoice(subscription.id)
    
    # Record usage
    billing.record_usage("cus_123", requests=100, tokens=5000)
    
    # Get current bill
    current_bill = billing.get_current_bill("cus_123")
    ```
    """
    
    def __init__(self, storage_path: str = "./.billing.json"):
        """
        Initialize billing manager.
        
        Args:
            storage_path: Path to store billing data.
        """
        self._storage_path = storage_path
        self._plans: Dict[str, Plan] = {}
        self._subscriptions: Dict[str, Subscription] = {}
        self._invoices: Dict[str, Invoice] = {}
        self._usage: List[UsageRecord] = []
        self._customers: Dict[str, Dict[str, Any]] = {}
        self._load_data()
        self._setup_default_plans()
    
    def _load_data(self):
        """Load billing data from storage."""
        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)
                
                for plan_data in data.get("plans", []):
                    plan = Plan(**plan_data)
                    self._plans[plan.id] = plan
                
                for sub_data in data.get("subscriptions", []):
                    sub_data["status"] = SubscriptionStatus(sub_data["status"])
                    sub_data["billing_cycle"] = BillingCycle(sub_data["billing_cycle"])
                    sub = Subscription(**sub_data)
                    self._subscriptions[sub.id] = sub
                
                for inv_data in data.get("invoices", []):
                    inv_data["status"] = PaymentStatus(inv_data["status"])
                    inv = Invoice(**inv_data)
                    self._invoices[inv.id] = inv
                
                for usage_data in data.get("usage", []):
                    usage = UsageRecord(**usage_data)
                    self._usage.append(usage)
                
                self._customers = data.get("customers", {})
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    def _save_data(self):
        """Save billing data to storage."""
        data = {
            "plans": [p.__dict__ for p in self._plans.values()],
            "subscriptions": [s.to_dict() for s in self._subscriptions.values()],
            "invoices": [i.to_dict() for i in self._invoices.values()],
            "usage": [u.to_dict() for u in self._usage],
            "customers": self._customers,
        }
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def _setup_default_plans(self):
        """Setup default subscription plans."""
        if not self._plans:
            self.create_plan(
                id="free",
                name="Free",
                description="For hobbyists and testing",
                price_monthly=0,
                price_yearly=0,
                features=["100 requests/day", "Basic models"],
                limits={"requests_per_day": 100, "requests_per_month": 1000},
            )
            
            self.create_plan(
                id="starter",
                name="Starter",
                description="For small projects",
                price_monthly=9.99,
                price_yearly=99.99,
                features=["1K requests/day", "All models", "Email support"],
                limits={"requests_per_day": 1000, "requests_per_month": 10000},
            )
            
            self.create_plan(
                id="pro",
                name="Pro",
                description="For production applications",
                price_monthly=29.99,
                price_yearly=299.99,
                features=["10K requests/day", "All models", "Priority support", "Webhooks"],
                limits={"requests_per_day": 10000, "requests_per_month": 100000},
            )
            
            self.create_plan(
                id="enterprise",
                name="Enterprise",
                description="For large scale deployments",
                price_monthly=99.99,
                price_yearly=999.99,
                features=["Unlimited requests", "Dedicated support", "SLA", "Custom integrations"],
                limits={"requests_per_day": -1, "requests_per_month": -1},
            )
    
    @staticmethod
    def generate_id(prefix: str) -> str:
        """Generate a unique ID."""
        return f"{prefix}_{secrets.token_hex(8)}"
    
    def create_plan(
        self,
        id: str,
        name: str,
        description: str,
        price_monthly: float,
        price_yearly: float,
        features: List[str],
        limits: Dict[str, Any],
    ) -> Plan:
        """Create a subscription plan."""
        plan = Plan(
            id=id,
            name=name,
            description=description,
            price_monthly=price_monthly,
            price_yearly=price_yearly,
            features=features,
            limits=limits,
        )
        self._plans[id] = plan
        self._save_data()
        return plan
    
    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get a plan by ID."""
        return self._plans.get(plan_id)
    
    def list_plans(self) -> List[Plan]:
        """List all active plans."""
        return [p for p in self._plans.values() if p.is_active]
    
    def create_customer(
        self,
        email: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a customer."""
        customer_id = self.generate_id("cus")
        customer = {
            "id": customer_id,
            "email": email,
            "name": name,
            "created_at": time.time(),
            "metadata": metadata or {},
        }
        self._customers[customer_id] = customer
        self._save_data()
        return customer
    
    def get_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get customer by ID."""
        return self._customers.get(customer_id)
    
    def create_subscription(
        self,
        customer_id: str,
        plan_id: str,
        billing_cycle: BillingCycle = BillingCycle.MONTHLY,
        trial_days: int = 0,
    ) -> Optional[Subscription]:
        """Create a new subscription."""
        plan = self._plans.get(plan_id)
        if not plan:
            return None
        
        subscription_id = self.generate_id("sub")
        now = time.time()
        
        period_days = {
            BillingCycle.MONTHLY: 30,
            BillingCycle.QUARTERLY: 90,
            BillingCycle.YEARLY: 365,
        }
        
        period_length = period_days[billing_cycle]
        
        subscription = Subscription(
            id=subscription_id,
            customer_id=customer_id,
            plan_id=plan_id,
            status=SubscriptionStatus.TRIALING if trial_days > 0 else SubscriptionStatus.ACTIVE,
            billing_cycle=billing_cycle,
            current_period_start=now,
            current_period_end=now + (period_length * 86400),
            trial_end=now + (trial_days * 86400) if trial_days > 0 else None,
        )
        
        self._subscriptions[subscription_id] = subscription
        self._save_data()
        return subscription
    
    def get_subscription(self, subscription_id: str) -> Optional[Subscription]:
        """Get subscription by ID."""
        return self._subscriptions.get(subscription_id)
    
    def get_customer_subscription(self, customer_id: str) -> Optional[Subscription]:
        """Get active subscription for customer."""
        for sub in self._subscriptions.values():
            if sub.customer_id == customer_id and sub.is_active():
                return sub
        return None
    
    def cancel_subscription(self, subscription_id: str) -> bool:
        """Cancel a subscription."""
        sub = self._subscriptions.get(subscription_id)
        if sub:
            sub.status = SubscriptionStatus.CANCELLED
            sub.cancelled_at = time.time()
            self._save_data()
            return True
        return False
    
    def change_plan(
        self,
        subscription_id: str,
        new_plan_id: str,
    ) -> Optional[Subscription]:
        """Change subscription plan."""
        sub = self._subscriptions.get(subscription_id)
        new_plan = self._plans.get(new_plan_id)
        
        if not sub or not new_plan:
            return None
        
        sub.plan_id = new_plan_id
        self._save_data()
        return sub
    
    def create_invoice(
        self,
        subscription_id: str,
        items: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Invoice]:
        """Create an invoice."""
        sub = self._subscriptions.get(subscription_id)
        if not sub:
            return None
        
        plan = self._plans.get(sub.plan_id)
        if not plan:
            return None
        
        amount = plan.get_price(sub.billing_cycle)
        
        period_length = {
            BillingCycle.MONTHLY: 30,
            BillingCycle.QUARTERLY: 90,
            BillingCycle.YEARLY: 365,
        }
        
        invoice = Invoice(
            id=self.generate_id("inv"),
            customer_id=sub.customer_id,
            subscription_id=subscription_id,
            amount=amount,
            period_start=sub.current_period_start,
            period_end=sub.current_period_end,
            items=items or [
                {"description": f"{plan.name} - {sub.billing_cycle.value}", "amount": amount}
            ],
        )
        
        self._invoices[invoice.id] = invoice
        self._save_data()
        return invoice
    
    def pay_invoice(self, invoice_id: str) -> bool:
        """Mark invoice as paid."""
        invoice = self._invoices.get(invoice_id)
        if invoice:
            invoice.status = PaymentStatus.SUCCEEDED
            invoice.paid_at = time.time()
            
            sub = self._subscriptions.get(invoice.subscription_id)
            if sub and sub.status == SubscriptionStatus.PAST_DUE:
                sub.status = SubscriptionStatus.ACTIVE
            
            self._save_data()
            return True
        return False
    
    def fail_invoice(self, invoice_id: str) -> bool:
        """Mark invoice as failed."""
        invoice = self._invoices.get(invoice_id)
        if invoice:
            invoice.status = PaymentStatus.FAILED
            
            sub = self._subscriptions.get(invoice.subscription_id)
            if sub:
                sub.status = SubscriptionStatus.PAST_DUE
            
            self._save_data()
            return True
        return False
    
    def record_usage(
        self,
        customer_id: str,
        key_id: str,
        requests: int = 0,
        tokens: int = 0,
    ) -> UsageRecord:
        """Record API usage."""
        record = UsageRecord(
            id=self.generate_id("use"),
            customer_id=customer_id,
            key_id=key_id,
            requests=requests,
            tokens_used=tokens,
            cost=self._calculate_cost(requests, tokens),
        )
        self._usage.append(record)
        self._save_data()
        return record
    
    def _calculate_cost(self, requests: int, tokens: int) -> float:
        """Calculate cost for usage."""
        cost_per_request = 0.001
        cost_per_token = 0.00001
        return (requests * cost_per_request) + (tokens * cost_per_token)
    
    def get_usage(
        self,
        customer_id: str,
        start_date: Optional[float] = None,
        end_date: Optional[float] = None,
    ) -> List[UsageRecord]:
        """Get usage records for customer."""
        records = [r for r in self._usage if r.customer_id == customer_id]
        
        if start_date:
            records = [r for r in records if r.timestamp >= start_date]
        if end_date:
            records = [r for r in records if r.timestamp <= end_date]
        
        return records
    
    def get_current_bill(self, customer_id: str) -> Dict[str, Any]:
        """Get current billing period summary."""
        sub = self.get_customer_subscription(customer_id)
        if not sub:
            return {"error": "No active subscription"}
        
        plan = self._plans.get(sub.plan_id)
        usage = self.get_usage(
            customer_id,
            start_date=sub.current_period_start,
            end_date=sub.current_period_end,
        )
        
        total_requests = sum(r.requests for r in usage)
        total_tokens = sum(r.tokens_used for r in usage)
        total_cost = sum(r.cost for r in usage)
        
        return {
            "subscription_id": sub.id,
            "plan": plan.to_dict() if plan else None,
            "billing_cycle": sub.billing_cycle.value,
            "period_start": sub.current_period_start,
            "period_end": sub.current_period_end,
            "remaining_days": sub.get_remaining_days(),
            "usage": {
                "total_requests": total_requests,
                "total_tokens": total_tokens,
                "total_cost": total_cost,
            },
            "limits": plan.limits if plan else {},
            "invoices": [
                inv.to_dict() for inv in self._invoices.values()
                if inv.subscription_id == sub.id
            ],
        }
    
    def get_invoices(self, customer_id: str) -> List[Invoice]:
        """Get all invoices for customer."""
        return [
            inv for inv in self._invoices.values()
            if inv.customer_id == customer_id
        ]
