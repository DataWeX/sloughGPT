import numpy as np
import pytest

from domains.collections.baby_economy import (
    ResourceType, TradeStatus, EconomyConfig, Resource, TradeOffer,
    BabyEconomy, MarketSystem,
)


class TestResource:
    def test_create(self):
        r = Resource(ResourceType.FOOD, 10.0, 0.8)
        assert r.amount == 10.0
        assert r.quality == 0.8

    def test_value(self):
        r = Resource(ResourceType.FOOD, 10.0, 0.8)
        assert r.value == 8.0

    def test_split(self):
        r = Resource(ResourceType.FOOD, 10.0, 0.8)
        r2 = r.split(3.0)
        assert r.amount == 7.0
        assert r2.amount == 3.0
        assert r2.quality == 0.8

    def test_split_too_much(self):
        r = Resource(ResourceType.FOOD, 5.0, 0.8)
        r2 = r.split(10.0)
        assert r.amount == 0.0
        assert r2.amount == 5.0

    def test_merge(self):
        r1 = Resource(ResourceType.FOOD, 5.0, 0.8)
        r2 = Resource(ResourceType.FOOD, 5.0, 0.6)
        r1.merge(r2)
        assert r1.amount == 10.0
        assert abs(r1.quality - 0.7) < 1e-6


class TestTradeOffer:
    def test_create(self):
        offer = TradeOffer(
            offer_id=0, seller_id=1, buyer_id=2,
            offer_resource=Resource(ResourceType.FOOD, 5.0),
            want_resource=Resource(ResourceType.WOOD, 3.0),
        )
        assert offer.status == TradeStatus.PENDING

    def test_accept(self):
        offer = TradeOffer(
            offer_id=0, seller_id=1, buyer_id=2,
            offer_resource=Resource(ResourceType.FOOD, 5.0),
            want_resource=Resource(ResourceType.WOOD, 3.0),
        )
        success = offer.accept()
        assert success
        assert offer.status == TradeStatus.ACCEPTED

    def test_reject(self):
        offer = TradeOffer(
            offer_id=0, seller_id=1, buyer_id=2,
            offer_resource=Resource(ResourceType.FOOD, 5.0),
            want_resource=Resource(ResourceType.WOOD, 3.0),
        )
        success = offer.reject()
        assert success
        assert offer.status == TradeStatus.REJECTED

    def test_complete(self):
        offer = TradeOffer(
            offer_id=0, seller_id=1, buyer_id=2,
            offer_resource=Resource(ResourceType.FOOD, 5.0),
            want_resource=Resource(ResourceType.WOOD, 3.0),
        )
        offer.accept()
        success = offer.complete()
        assert success
        assert offer.status == TradeStatus.COMPLETED

    def test_cancel(self):
        offer = TradeOffer(
            offer_id=0, seller_id=1, buyer_id=2,
            offer_resource=Resource(ResourceType.FOOD, 5.0),
            want_resource=Resource(ResourceType.WOOD, 3.0),
        )
        success = offer.cancel()
        assert success
        assert offer.status == TradeStatus.CANCELLED


class TestBabyEconomy:
    def test_add_resource(self):
        econ = BabyEconomy(1)
        r = Resource(ResourceType.FOOD, 10.0)
        added = econ.add_resource(r)
        assert added == 10.0
        assert econ.get_resource(ResourceType.FOOD).amount == 10.0

    def test_add_resource_overflow(self):
        config = EconomyConfig(max_inventory=20)
        econ = BabyEconomy(1, config)
        econ.add_resource(Resource(ResourceType.FOOD, 15.0))
        econ.add_resource(Resource(ResourceType.WOOD, 10.0))
        assert econ.get_resource(ResourceType.WOOD).amount == 5.0

    def test_remove_resource(self):
        econ = BabyEconomy(1)
        econ.add_resource(Resource(ResourceType.FOOD, 10.0))
        removed = econ.remove_resource(ResourceType.FOOD, 3.0)
        assert removed is not None
        assert removed.amount == 3.0
        assert econ.get_resource(ResourceType.FOOD).amount == 7.0

    def test_remove_resource_not_enough(self):
        econ = BabyEconomy(1)
        econ.add_resource(Resource(ResourceType.FOOD, 5.0))
        removed = econ.remove_resource(ResourceType.FOOD, 10.0)
        assert removed is None

    def test_remove_resource_not_found(self):
        econ = BabyEconomy(1)
        removed = econ.remove_resource(ResourceType.FOOD, 5.0)
        assert removed is None

    def test_get_inventory_value(self):
        econ = BabyEconomy(1)
        econ.add_resource(Resource(ResourceType.FOOD, 10.0, 0.8))
        econ.add_resource(Resource(ResourceType.WOOD, 5.0, 1.0))
        value = econ.get_inventory_value()
        assert value == 13.0

    def test_create_offer(self):
        econ = BabyEconomy(1)
        econ.add_resource(Resource(ResourceType.FOOD, 10.0))
        offer = econ.create_offer(2, Resource(ResourceType.FOOD, 5.0), Resource(ResourceType.WOOD, 3.0))
        assert offer is not None
        assert offer.seller_id == 1
        assert offer.status == TradeStatus.PENDING

    def test_create_offer_insufficient(self):
        econ = BabyEconomy(1)
        econ.add_resource(Resource(ResourceType.FOOD, 2.0))
        offer = econ.create_offer(2, Resource(ResourceType.FOOD, 5.0), Resource(ResourceType.WOOD, 3.0))
        assert offer is None

    def test_accept_offer(self):
        seller = BabyEconomy(1)
        seller.add_resource(Resource(ResourceType.FOOD, 10.0))
        offer = seller.create_offer(2, Resource(ResourceType.FOOD, 5.0), Resource(ResourceType.WOOD, 3.0))

        buyer = BabyEconomy(2)
        buyer.add_resource(Resource(ResourceType.WOOD, 10.0))
        success = buyer.execute_trade_with(offer, seller)
        assert success
        assert buyer.get_resource(ResourceType.FOOD) is not None
        assert seller.get_resource(ResourceType.FOOD).amount == 5.0
        assert seller.get_resource(ResourceType.WOOD).amount == 3.0

    def test_price_history(self):
        econ = BabyEconomy(1)
        econ._update_price(ResourceType.FOOD, 10.0)
        econ._update_price(ResourceType.FOOD, 12.0)
        avg = econ.get_average_price(ResourceType.FOOD)
        assert abs(avg - 11.0) < 1e-6

    def test_price_trend(self):
        econ = BabyEconomy(1)
        for i in range(20):
            econ._update_price(ResourceType.FOOD, 10.0 + i * 0.5)
        trend = econ.get_price_trend(ResourceType.FOOD)
        assert trend > 0

    def test_summary(self):
        econ = BabyEconomy(1)
        econ.add_resource(Resource(ResourceType.FOOD, 10.0))
        s = econ.summary()
        assert s["baby_id"] == 1
        assert s["inventory_items"] == 1


class TestMarketSystem:
    def test_register(self):
        market = MarketSystem()
        econ = BabyEconomy(1)
        market.register(1, econ)
        assert 1 in market._economies

    def test_unregister(self):
        market = MarketSystem()
        econ = BabyEconomy(1)
        market.register(1, econ)
        market.unregister(1)
        assert 1 not in market._economies

    def test_list_offers(self):
        market = MarketSystem()
        econ1 = BabyEconomy(1)
        econ1.add_resource(Resource(ResourceType.FOOD, 10.0))
        econ1.create_offer(2, Resource(ResourceType.FOOD, 5.0), Resource(ResourceType.WOOD, 3.0))
        market.register(1, econ1)
        offers = market.list_offers()
        assert len(offers) == 1

    def test_find_trade(self):
        market = MarketSystem()
        seller = BabyEconomy(1)
        seller.add_resource(Resource(ResourceType.FOOD, 10.0))
        seller.create_offer(2, Resource(ResourceType.FOOD, 5.0), Resource(ResourceType.WOOD, 3.0))
        market.register(1, seller)
        offer = market.find_trade(2, ResourceType.WOOD, ResourceType.FOOD)
        assert offer is not None

    def test_execute_trade(self):
        market = MarketSystem()
        seller = BabyEconomy(1)
        seller.add_resource(Resource(ResourceType.FOOD, 10.0))
        offer = seller.create_offer(2, Resource(ResourceType.FOOD, 5.0), Resource(ResourceType.WOOD, 3.0))
        market.register(1, seller)

        buyer = BabyEconomy(2)
        buyer.add_resource(Resource(ResourceType.WOOD, 10.0))
        market.register(2, buyer)

        success = market.execute_trade(offer, buyer)
        assert success
        assert buyer.get_resource(ResourceType.FOOD) is not None

    def test_market_stats(self):
        market = MarketSystem()
        econ1 = BabyEconomy(1)
        econ2 = BabyEconomy(2)
        market.register(1, econ1)
        market.register(2, econ2)
        stats = market.get_market_stats()
        assert stats["participants"] == 2

    def test_summary(self):
        market = MarketSystem()
        econ = BabyEconomy(1)
        market.register(1, econ)
        s = market.summary()
        assert "market_stats" in s
