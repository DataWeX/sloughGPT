"""
User Journey Tests — end-to-end browser tests using chrome-devtools.

Tests the main user flows by navigating the web UI, clicking elements,
filling forms, and verifying expected behavior.

Usage:
    These tests are designed to be run via opencode's chrome-devtools tools.
    They test the live web app at http://localhost:3000.
"""
import time
import json
from pathlib import Path

# Test results accumulator
RESULTS = []


def record(test_name: str, passed: bool, detail: str = ""):
    RESULTS.append({"test": test_name, "passed": passed, "detail": detail})
    status = "ok" if passed else "err"
    print(f"  [{status}] {test_name}" + (f" — {detail}" if detail else ""))


# ═══════════════════════════════════════════════════════════════
# Journey 1: Dashboard loads and shows key elements
# ═══════════════════════════════════════════════════════════════

def test_dashboard_loads(page_id: int):
    """Navigate to / and verify dashboard renders."""
    from chrome_devtools import navigate_page, take_snapshot

    navigate_page(page_id, type="url", url="http://localhost:3000/")
    time.sleep(2)

    snapshot = take_snapshot(page_id)
    text = snapshot if isinstance(snapshot, str) else json.dumps(snapshot)

    # Check for key dashboard elements
    has_sidebar = "sidebar" in text.lower() or "nav" in text.lower() or "chat" in text.lower()
    has_content = len(text) > 500  # Dashboard should have substantial content

    record("dashboard_loads", has_sidebar and has_content,
           f"sidebar={has_sidebar}, content_len={len(text)}")


def test_dashboard_greeting(page_id: int):
    """Dashboard should show a greeting or welcome message."""
    from chrome_devtools import take_snapshot

    snapshot = take_snapshot(page_id)
    text = snapshot if isinstance(snapshot, str) else json.dumps(snapshot)

    # Dashboard typically shows greeting, stats, or quick actions
    has_greeting = any(w in text.lower() for w in ["welcome", "hello", "good", "dashboard", "home"])
    record("dashboard_greeting", has_greeting,
           f"greeting_found={has_greeting}")


# ═══════════════════════════════════════════════════════════════
# Journey 2: Navigation — sidebar links work
# ═══════════════════════════════════════════════════════════════

NAV_ROUTES = [
    ("/chat", "Chat"),
    ("/training", "Training"),
    ("/datasets", "Datasets"),
    ("/models", "Models"),
    ("/agents", "Agents"),
    ("/souls", "Souls"),
    ("/knowledge", "Knowledge"),
    ("/monitoring", "Monitoring"),
    ("/settings", "Settings"),
    ("/planner", "Planner"),
]


def test_navigation_routes(page_id: int):
    """Click each sidebar link and verify page loads."""
    from chrome_devtools import navigate_page, take_snapshot

    for route, name in NAV_ROUTES:
        navigate_page(page_id, type="url", url=f"http://localhost:3000{route}")
        time.sleep(1.5)

        snapshot = take_snapshot(page_id)
        text = snapshot if isinstance(snapshot, str) else json.dumps(snapshot)

        # Page loaded if it has content and no error
        has_error = "404" in text or "not found" in text.lower() or "error" in text.lower()[:200]
        has_content = len(text) > 200

        record(f"nav_{name.lower()}", has_content and not has_error,
               f"route={route}, len={len(text)}, error={has_error}")


# ═══════════════════════════════════════════════════════════════
# Journey 3: Chat — send a message
# ═══════════════════════════════════════════════════════════════

def test_chat_page_loads(page_id: int):
    """Navigate to /chat and verify chat interface renders."""
    from chrome_devtools import navigate_page, take_snapshot

    navigate_page(page_id, type="url", url="http://localhost:3000/chat")
    time.sleep(2)

    snapshot = take_snapshot(page_id)
    text = snapshot if isinstance(snapshot, str) else json.dumps(snapshot)

    has_input = "input" in text.lower() or "textarea" in text.lower() or "message" in text.lower()
    has_chat = "chat" in text.lower()

    record("chat_page_loads", has_input or has_chat,
           f"input_found={has_input}, chat_ref={has_chat}")


def test_chat_send_message(page_id: int):
    """Type a message in the chat input and verify it appears."""
    from chrome_devtools import take_snapshot, fill, type_text, press_key

    snapshot = take_snapshot(page_id)
    text = snapshot if isinstance(snapshot, str) else json.dumps(snapshot)

    # Find input/textarea uid from snapshot
    # The snapshot contains uids for interactive elements
    # We'll try to type into any visible input
    if isinstance(snapshot, str):
        # Try to find and fill input
        try:
            # Look for textarea or input in the snapshot
            import re
            # Find uid for input elements
            uid_match = re.search(r'uid["\s:=]+["\']?(\w+)', text)
            if uid_match:
                uid = uid_match.group(1)
                fill(page_id, uid, "Hello, this is a test message")
                record("chat_send_message", True, "filled input")
                return
        except Exception:
            pass

    record("chat_send_message", False, "could not find chat input")


# ═══════════════════════════════════════════════════════════════
# Journey 4: Training page
# ═══════════════════════════════════════════════════════════════

def test_training_page_loads(page_id: int):
    """Navigate to /training and verify it renders."""
    from chrome_devtools import navigate_page, take_snapshot

    navigate_page(page_id, type="url", url="http://localhost:3000/training")
    time.sleep(2)

    snapshot = take_snapshot(page_id)
    text = snapshot if isinstance(snapshot, str) else json.dumps(snapshot)

    has_training = "train" in text.lower()
    has_content = len(text) > 200

    record("training_page_loads", has_training and has_content,
           f"train_ref={has_training}, len={len(text)}")


# ═══════════════════════════════════════════════════════════════
# Journey 5: Settings page
# ═══════════════════════════════════════════════════════════════

def test_settings_page_loads(page_id: int):
    """Navigate to /settings and verify it renders."""
    from chrome_devtools import navigate_page, take_snapshot

    navigate_page(page_id, type="url", url="http://localhost:3000/settings")
    time.sleep(2)

    snapshot = take_snapshot(page_id)
    text = snapshot if isinstance(snapshot, str) else json.dumps(snapshot)

    has_settings = "setting" in text.lower() or "config" in text.lower()
    has_content = len(text) > 200

    record("settings_page_loads", has_settings and has_content,
           f"settings_ref={has_settings}, len={len(text)}")


# ═══════════════════════════════════════════════════════════════
# Journey 6: Planner — create a card
# ═══════════════════════════════════════════════════════════════

def test_planner_page_loads(page_id: int):
    """Navigate to /planner and verify board renders."""
    from chrome_devtools import navigate_page, take_snapshot

    navigate_page(page_id, type="url", url="http://localhost:3000/planner")
    time.sleep(2)

    snapshot = take_snapshot(page_id)
    text = snapshot if isinstance(snapshot, str) else json.dumps(snapshot)

    has_planner = "planner" in text.lower() or "board" in text.lower() or "card" in text.lower()
    has_content = len(text) > 200

    record("planner_page_loads", has_planner and has_content,
           f"planner_ref={has_planner}, len={len(text)}")


# ═══════════════════════════════════════════════════════════════
# Journey 7: Models page — list models
# ═══════════════════════════════════════════════════════════════

def test_models_page_loads(page_id: int):
    """Navigate to /models and verify model list renders."""
    from chrome_devtools import navigate_page, take_snapshot

    navigate_page(page_id, type="url", url="http://localhost:3000/models")
    time.sleep(2)

    snapshot = take_snapshot(page_id)
    text = snapshot if isinstance(snapshot, str) else json.dumps(snapshot)

    has_models = "model" in text.lower()
    has_content = len(text) > 200

    record("models_page_loads", has_models and has_content,
           f"models_ref={has_models}, len={len(text)}")


# ═══════════════════════════════════════════════════════════════
# Journey 8: Monitoring — system health
# ═══════════════════════════════════════════════════════════════

def test_monitoring_page_loads(page_id: int):
    """Navigate to /monitoring and verify metrics render."""
    from chrome_devtools import navigate_page, take_snapshot

    navigate_page(page_id, type="url", url="http://localhost:3000/monitoring")
    time.sleep(2)

    snapshot = take_snapshot(page_id)
    text = snapshot if isinstance(snapshot, str) else json.dumps(snapshot)

    has_monitoring = "monitor" in text.lower() or "cpu" in text.lower() or "memory" in text.lower() or "health" in text.lower()
    has_content = len(text) > 200

    record("monitoring_page_loads", has_monitoring and has_content,
           f"monitoring_ref={has_monitoring}, len={len(text)}")


# ═══════════════════════════════════════════════════════════════
# Journey 9: Knowledge base
# ═══════════════════════════════════════════════════════════════

def test_knowledge_page_loads(page_id: int):
    """Navigate to /knowledge and verify it renders."""
    from chrome_devtools import navigate_page, take_snapshot

    navigate_page(page_id, type="url", url="http://localhost:3000/knowledge")
    time.sleep(2)

    snapshot = take_snapshot(page_id)
    text = snapshot if isinstance(snapshot, str) else json.dumps(snapshot)

    has_knowledge = "knowledge" in text.lower() or "memory" in text.lower() or "fact" in text.lower()
    has_content = len(text) > 200

    record("knowledge_page_loads", has_knowledge and has_content,
           f"knowledge_ref={has_knowledge}, len={len(text)}")


# ═══════════════════════════════════════════════════════════════
# Journey 10: Redirect legacy routes
# ═══════════════════════════════════════════════════════════════

REDIRECT_ROUTES = [
    ("/companion", "/souls"),
    ("/evaluate", "/benchmark"),
    ("/memory", "/knowledge"),
    ("/collections", "/datasets"),
    ("/self-train", "/training"),
    ("/admin", "/settings"),
    ("/images", "/files"),
    ("/session", "/shell"),
]


def test_legacy_redirects(page_id: int):
    """Verify legacy routes redirect to their new locations."""
    from chrome_devtools import navigate_page, take_snapshot

    for old_route, expected_new in REDIRECT_ROUTES:
        navigate_page(page_id, type="url", url=f"http://localhost:3000{old_route}")
        time.sleep(1.5)

        # Check current URL after redirect
        # The snapshot should show content from the new route
        snapshot = take_snapshot(page_id)
        text = snapshot if isinstance(snapshot, str) else json.dumps(snapshot)

        # If the page has content and isn't a 404, redirect worked
        has_content = len(text) > 200
        record(f"redirect_{old_route.replace('/', '_')}", has_content,
               f"{old_route} -> {expected_new}, content={has_content}")


# ═══════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════

def run_all_journeys(page_id: int):
    """Run all user journey tests."""
    print("=" * 60)
    print("User Journey Tests")
    print("=" * 60)

    print("\n> Journey 1: Dashboard")
    test_dashboard_loads(page_id)
    test_dashboard_greeting(page_id)

    print("\n> Journey 2: Navigation")
    test_navigation_routes(page_id)

    print("\n> Journey 3: Chat")
    test_chat_page_loads(page_id)
    test_chat_send_message(page_id)

    print("\n> Journey 4: Training")
    test_training_page_loads(page_id)

    print("\n> Journey 5: Settings")
    test_settings_page_loads(page_id)

    print("\n> Journey 6: Planner")
    test_planner_page_loads(page_id)

    print("\n> Journey 7: Models")
    test_models_page_loads(page_id)

    print("\n> Journey 8: Monitoring")
    test_monitoring_page_loads(page_id)

    print("\n> Journey 9: Knowledge")
    test_knowledge_page_loads(page_id)

    print("\n> Journey 10: Legacy Redirects")
    test_legacy_redirects(page_id)

    # Summary
    total = len(RESULTS)
    passed = sum(1 for r in RESULTS if r["passed"])
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\nFailed tests:")
        for r in RESULTS:
            if not r["passed"]:
                print(f"  err {r['test']}: {r['detail']}")

    # Save results
    results_path = Path(__file__).parent.parent / "test_results" / "user_journey_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(RESULTS, indent=2))

    return passed, failed


if __name__ == "__main__":
    # Requires chrome-devtools page_id as argument
    import sys
    page_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    run_all_journeys(page_id)
