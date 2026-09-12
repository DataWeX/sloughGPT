from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import IntEnum


class ResourceType(IntEnum):
    FOOD = 1
    WOOD = 2
    STONE = 3
    ENERGY = 4
    KNOWLEDGE = 5
    TOOL = 6
    MATERIAL = 7
    CURRENCY = 8


class TradeStatus(IntEnum):
    PENDING = 1
    ACCEPTED = 2
    REJECTED = 3
    COMPLETED = 4
    CANCELLED = 5


@dataclass
class EconomyConfig:
    max_inventory: int = 100
    trade_range: float = 5.0
    min_trade_value: float = 0.1
    tax_rate: float = 0.05
    inflation_rate: float = 0.01
    max_offers: int = 10
    price_history_size: int = 50


@dataclass
class Resource:
    resource_type: ResourceType
    amount: float = 0.0
    quality: float = 1.0
    metadata: dict = field(default_factory=dict)

    def split(self, amount: float) -> 'Resource':
        actual = min(amount, self.amount)
        self.amount -= actual
        return Resource(self.resource_type, actual, self.quality, dict(self.metadata))

    def merge(self, other: 'Resource'):
        if self.resource_type == other.resource_type:
            total = self.amount + other.amount
            self.quality = (self.quality * self.amount + other.quality * other.amount) / max(total, 1e-8)
            self.amount = total

    @property
    def value(self) -> float:
        return self.amount * self.quality


@dataclass
class TradeOffer:
    offer_id: int
    seller_id: int
    buyer_id: int | None
    offer_resource: Resource
    want_resource: Resource
    status: TradeStatus = TradeStatus.PENDING
    created_tick: int = 0
    metadata: dict = field(default_factory=dict)

    def accept(self) -> bool:
        if self.status != TradeStatus.PENDING:
            return False
        self.status = TradeStatus.ACCEPTED
        return True

    def reject(self) -> bool:
        if self.status != TradeStatus.PENDING:
            return False
        self.status = TradeStatus.REJECTED
        return True

    def complete(self) -> bool:
        if self.status != TradeStatus.ACCEPTED:
            return False
        self.status = TradeStatus.COMPLETED
        return True

    def cancel(self) -> bool:
        if self.status in (TradeStatus.COMPLETED, TradeStatus.REJECTED):
            return False
        self.status = TradeStatus.CANCELLED
        return True


class BabyEconomy:
    def __init__(self, baby_id: int, config: EconomyConfig | None = None):
        self.baby_id = baby_id
        self.config = config or EconomyConfig()
        self._inventory: dict[ResourceType, Resource] = {}
        self._wallet: float = 0.0
        self._trade_history: list[TradeOffer] = []
        self._active_offers: list[TradeOffer] = []
        self._price_history: dict[ResourceType, list[float]] = {}
        self._next_offer_id = 0

    def add_resource(self, resource: Resource) -> float:
        total = sum(r.amount for r in self._inventory.values())
        if total + resource.amount > self.config.max_inventory:
            space = self.config.max_inventory - total
            actual = min(resource.amount, space)
            resource = Resource(resource.resource_type, actual, resource.quality)
        if resource.resource_type in self._inventory:
            self._inventory[resource.resource_type].merge(resource)
        else:
            self._inventory[resource.resource_type] = resource
        return resource.amount

    def remove_resource(self, resource_type: ResourceType, amount: float) -> Resource | None:
        if resource_type not in self._inventory:
            return None
        current = self._inventory[resource_type]
        if current.amount < amount:
            return None
        removed = current.split(amount)
        if current.amount <= 0:
            del self._inventory[resource_type]
        return removed

    def get_resource(self, resource_type: ResourceType) -> Resource | None:
        return self._inventory.get(resource_type)

    def get_inventory_value(self) -> float:
        return sum(r.value for r in self._inventory.values())

    def get_inventory_summary(self) -> dict:
        return {
            rt.name: {"amount": r.amount, "quality": r.quality, "value": r.value}
            for rt, r in self._inventory.items()
        }

    def create_offer(self, buyer_id: int | None, offer_resource: Resource,
                     want_resource: Resource) -> TradeOffer | None:
        if len(self._active_offers) >= self.config.max_offers:
            return None
        if self.get_resource(offer_resource.resource_type) is None:
            return None
        if self.get_resource(offer_resource.resource_type).amount < offer_resource.amount:
            return None
        offer = TradeOffer(
            offer_id=self._next_offer_id,
            seller_id=self.baby_id,
            buyer_id=buyer_id,
            offer_resource=offer_resource,
            want_resource=want_resource,
        )
        self._active_offers.append(offer)
        self._next_offer_id += 1
        return offer

    def accept_offer(self, offer: TradeOffer) -> bool:
        if offer.buyer_id != self.baby_id:
            return False
        if not offer.accept():
            return False
        return True

    def execute_trade_with(self, offer: TradeOffer, other_economy: 'BabyEconomy') -> bool:
        if offer.status == TradeStatus.PENDING:
            offer.accept()
        if offer.status != TradeStatus.ACCEPTED:
            return False
        seller = other_economy if offer.seller_id == other_economy.baby_id else self
        buyer = self if offer.buyer_id == self.baby_id else other_economy
        seller_res = seller.get_resource(offer.offer_resource.resource_type)
        if seller_res is None or seller_res.amount < offer.offer_resource.amount:
            offer.cancel()
            return False
        buyer_res = buyer.get_resource(offer.want_resource.resource_type)
        if buyer_res is None or buyer_res.amount < offer.want_resource.amount:
            offer.cancel()
            return False
        seller.remove_resource(offer.offer_resource.resource_type, offer.offer_resource.amount)
        buyer.remove_resource(offer.want_resource.resource_type, offer.want_resource.amount)
        buyer.add_resource(offer.offer_resource)
        seller.add_resource(offer.want_resource)
        self._update_price(offer.offer_resource.resource_type, offer.offer_resource.value)
        self._trade_history.append(offer)
        if offer in self._active_offers:
            self._active_offers.remove(offer)
        offer.complete()
        return True

    def _execute_trade(self, offer: TradeOffer) -> bool:
        seller_res = self.get_resource(offer.offer_resource.resource_type)
        if seller_res is None or seller_res.amount < offer.offer_resource.amount:
            offer.cancel()
            return False
        self.remove_resource(offer.offer_resource.resource_type, offer.offer_resource.amount)
        self.add_resource(offer.want_resource)
        self._update_price(offer.offer_resource.resource_type, offer.offer_resource.value)
        self._trade_history.append(offer)
        if offer in self._active_offers:
            self._active_offers.remove(offer)
        offer.complete()
        return True

    def _update_price(self, resource_type: ResourceType, price: float):
        if resource_type not in self._price_history:
            self._price_history[resource_type] = []
        self._price_history[resource_type].append(price)
        if len(self._price_history[resource_type]) > self.config.price_history_size:
            self._price_history[resource_type] = self._price_history[resource_type][-self.config.price_history_size:]

    def get_average_price(self, resource_type: ResourceType) -> float:
        prices = self._price_history.get(resource_type, [])
        if not prices:
            return 0.0
        return float(np.mean(prices))

    def get_price_trend(self, resource_type: ResourceType) -> float:
        prices = self._price_history.get(resource_type, [])
        if len(prices) < 2:
            return 0.0
        recent = np.mean(prices[-10:])
        older = np.mean(prices[:-10]) if len(prices) > 10 else prices[0]
        return float(recent - older)

    def tick(self):
        for offer in self._active_offers:
            if offer.status == TradeStatus.PENDING:
                offer.metadata["ticks_alive"] = offer.metadata.get("ticks_alive", 0) + 1

    def summary(self) -> dict:
        return {
            "baby_id": self.baby_id,
            "inventory_value": self.get_inventory_value(),
            "inventory_items": len(self._inventory),
            "active_offers": len(self._active_offers),
            "trade_history": len(self._trade_history),
        }


class MarketSystem:
    def __init__(self, config: EconomyConfig | None = None):
        self.config = config or EconomyConfig()
        self._economies: dict[int, BabyEconomy] = {}
        self._global_offers: list[TradeOffer] = []
        self._market_history: list[dict] = []

    def register(self, baby_id: int, economy: BabyEconomy):
        self._economies[baby_id] = economy

    def unregister(self, baby_id: int):
        self._economies.pop(baby_id, None)

    def get_economy(self, baby_id: int) -> BabyEconomy | None:
        return self._economies.get(baby_id)

    def list_offers(self, resource_type: ResourceType | None = None) -> list[TradeOffer]:
        offers = []
        for economy in self._economies.values():
            for offer in economy._active_offers:
                if offer.status != TradeStatus.PENDING:
                    continue
                if resource_type and offer.offer_resource.resource_type != resource_type:
                    continue
                offers.append(offer)
        return offers

    def find_trade(self, buyer_id: int, want_type: ResourceType,
                   offer_type: ResourceType) -> TradeOffer | None:
        for economy in self._economies.values():
            if economy.baby_id == buyer_id:
                continue
            for offer in economy._active_offers:
                if offer.status != TradeStatus.PENDING:
                    continue
                if offer.offer_resource.resource_type == offer_type:
                    if offer.want_resource.resource_type == want_type:
                        return offer
        return None

    def execute_trade(self, offer: TradeOffer, buyer_economy: BabyEconomy) -> bool:
        seller_economy = self._economies.get(offer.seller_id)
        if seller_economy is None:
            return False
        if not offer.accept():
            return False
        return buyer_economy.execute_trade_with(offer, seller_economy)

    def tick(self):
        for economy in self._economies.values():
            economy.tick()

    def get_market_stats(self) -> dict:
        total_offers = sum(len(e._active_offers) for e in self._economies.values())
        total_trades = sum(len(e._trade_history) for e in self._economies.values())
        total_value = sum(e.get_inventory_value() for e in self._economies.values())
        return {
            "participants": len(self._economies),
            "active_offers": total_offers,
            "total_trades": total_trades,
            "total_value": total_value,
        }

    def summary(self) -> dict:
        return {
            "market_stats": self.get_market_stats(),
            "economies": len(self._economies),
        }
