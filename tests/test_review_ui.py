"""
Playwright tests for the Screenshot Review (Tinder-style classification) system.

Tests both desktop and mobile viewports for:
- Modal open/close
- Swipe gestures (right/left)
- Card stack rendering
- API integration (approve/classify/undo)
- Visual feedback (stamps, glow)
- Queue advancement
- Mobile layout (tab overflow, card sizing)

Usage:
    python3 tests/test_review_ui.py
"""

import json
import os
import sys
import time
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False
    print("WARNING: playwright not installed, skipping UI tests")

BASE_URL = "http://localhost:80"
SCREENSHOT_DIR = Path(__file__).parent / "screenshots" / "test_outputs"


def save_screenshot(page, name):
    """Save a debug screenshot."""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(SCREENSHOT_DIR / f"{name}.png"))


def swipe_card(page, direction='right', distance=150):
    """Simulate a swipe gesture on the active review card."""
    card = page.query_selector('.review-card:not(.stack-1):not(.stack-2)')
    if not card:
        return False
    header = card.query_selector('.review-card-header') or card
    box = header.bounding_box()
    if not box:
        return False
    cx = box['x'] + box['width'] / 2
    cy = box['y'] + box['height'] / 2

    page.mouse.move(cx, cy)
    page.mouse.down()
    dx = distance if direction == 'right' else -distance
    for step in range(0, abs(dx), 10):
        page.mouse.move(cx + (step if dx > 0 else -step), cy)
    page.mouse.up()
    return True


def open_review_modal(page, category_index=0):
    """Open the review modal for a category by clicking the review button.

    Waits on selectors rather than fixed sleeps: on a loaded box (OCR/VLM
    inference running) the Screenshots tab can take >800ms to render, and a
    fixed sleep made every test using this helper flaky.
    """
    page.click('text=Screenshots')
    try:
        page.wait_for_selector('.review-btn', timeout=10000)
    except Exception:
        return False
    review_btns = page.query_selector_all('.review-btn')
    if len(review_btns) > category_index:
        review_btns[category_index].click()
        try:
            page.wait_for_selector('#review-modal.open', timeout=10000)
        except Exception:
            return False
        page.wait_for_timeout(700)  # let the card stack render/settle
        return True
    return False


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright not installed")
class TestReviewModalDesktop(unittest.TestCase):
    """Test review modal on desktop viewport (1280x900)."""

    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.page = self.browser.new_page(viewport={'width': 1280, 'height': 900})
        self.page.goto(BASE_URL)
        self.page.wait_for_timeout(2000)

    def tearDown(self):
        self.page.close()

    def test_modal_opens_and_closes(self):
        """Review modal opens when clicking review button and closes on X."""
        open_review_modal(self.page, 0)  # Ads

        modal = self.page.query_selector('#review-modal')
        self.assertIsNotNone(modal)
        has_open = self.page.evaluate(
            'document.getElementById("review-modal").classList.contains("open")'
        )
        self.assertTrue(has_open, "Modal should be open")

        # Close with X button
        self.page.click('.review-close-btn')
        self.page.wait_for_timeout(400)
        has_open = self.page.evaluate(
            'document.getElementById("review-modal").classList.contains("open")'
        )
        self.assertFalse(has_open, "Modal should be closed")

    def test_modal_closes_on_escape(self):
        """Modal closes when pressing Escape."""
        open_review_modal(self.page, 0)
        self.page.keyboard.press('Escape')
        self.page.wait_for_timeout(400)
        has_open = self.page.evaluate(
            'document.getElementById("review-modal").classList.contains("open")'
        )
        self.assertFalse(has_open)

    def test_card_stack_renders(self):
        """3-card stack renders with active card and stacked cards behind."""
        open_review_modal(self.page, 0)

        cards = self.page.query_selector_all('.review-card')
        self.assertGreaterEqual(len(cards), 1, "Should have at least 1 card")
        self.assertLessEqual(len(cards), 3, "Should have at most 3 cards")

        # Active card should not have stack class
        active = self.page.query_selector('.review-card:not(.stack-1):not(.stack-2)')
        self.assertIsNotNone(active, "Should have an active card")

        save_screenshot(self.page, "desktop_card_stack")

    def test_card_has_image(self):
        """Active card displays a screenshot image."""
        open_review_modal(self.page, 0)

        img = self.page.query_selector('.review-card:not(.stack-1):not(.stack-2) .review-card-image img')
        self.assertIsNotNone(img, "Card should have an image")
        src = img.get_attribute('src')
        self.assertIn('/api/screenshots/', src, "Image src should point to screenshot API")

    def test_counter_displays(self):
        """Counter shows current position (e.g., '1 of 200')."""
        open_review_modal(self.page, 0)

        counter = self.page.query_selector('#review-counter')
        text = counter.inner_text()
        self.assertRegex(text, r'\d+ of \d+', f"Counter should show 'X of Y', got: {text}")

    def test_swipe_right_approves(self):
        """Swiping right on Ads category approves the screenshot."""
        open_review_modal(self.page, 0)

        counter_before = self.page.query_selector('#review-counter').inner_text()
        swipe_card(self.page, 'right', 150)
        self.page.wait_for_timeout(600)

        counter_after = self.page.query_selector('#review-counter').inner_text()
        self.assertNotEqual(counter_before, counter_after, "Counter should advance after swipe")
        save_screenshot(self.page, "desktop_after_swipe_right")

    def test_swipe_left_reclassifies(self):
        """Swiping left reclassifies the screenshot."""
        open_review_modal(self.page, 0)

        swipe_card(self.page, 'left', 150)
        self.page.wait_for_timeout(600)

        # Should have advanced
        counter = self.page.query_selector('#review-counter').inner_text()
        self.assertIn('of', counter)
        save_screenshot(self.page, "desktop_after_swipe_left")

    def test_insufficient_swipe_snaps_back(self):
        """Dragging less than threshold snaps card back to center."""
        open_review_modal(self.page, 0)

        counter_before = self.page.query_selector('#review-counter').inner_text()
        swipe_card(self.page, 'right', 30)  # Too short
        self.page.wait_for_timeout(500)

        counter_after = self.page.query_selector('#review-counter').inner_text()
        self.assertEqual(counter_before, counter_after, "Counter should NOT advance on insufficient swipe")

    def test_keyboard_arrow_right(self):
        """Arrow right key triggers approve action."""
        open_review_modal(self.page, 0)

        counter_before = self.page.query_selector('#review-counter').inner_text()
        self.page.keyboard.press('ArrowRight')
        self.page.wait_for_timeout(600)

        counter_after = self.page.query_selector('#review-counter').inner_text()
        self.assertNotEqual(counter_before, counter_after, "Arrow right should advance")

    def test_keyboard_arrow_left(self):
        """Arrow left key triggers reclassify action."""
        open_review_modal(self.page, 0)

        counter_before = self.page.query_selector('#review-counter').inner_text()
        self.page.keyboard.press('ArrowLeft')
        self.page.wait_for_timeout(600)

        counter_after = self.page.query_selector('#review-counter').inner_text()
        self.assertNotEqual(counter_before, counter_after, "Arrow left should advance")

    def test_undo_reverses_action(self):
        """Undo button reverses the last swipe action."""
        open_review_modal(self.page, 0)

        # Swipe right to approve
        self.page.keyboard.press('ArrowRight')
        self.page.wait_for_timeout(600)
        counter_after_swipe = self.page.query_selector('#review-counter').inner_text()

        # Undo. The undo server call is queued behind the swipe's POST, so
        # under load (e.g. VLM model loading) it can take >600ms — poll for
        # the counter change instead of a fixed sleep.
        self.page.click('.review-undo-btn')
        counter_after_undo = counter_after_swipe
        for _ in range(20):
            self.page.wait_for_timeout(250)
            counter_after_undo = self.page.query_selector('#review-counter').inner_text()
            if counter_after_undo != counter_after_swipe:
                break

        self.assertNotEqual(counter_after_swipe, counter_after_undo, "Undo should reverse position")

    def test_category_labels_correct(self):
        """Each category shows correct stamp labels."""
        # Ads: right=CORRECT, left=NOT AD
        open_review_modal(self.page, 0)
        right_hint = self.page.query_selector('#hint-right-label').inner_text()
        left_hint = self.page.query_selector('#hint-left-label').inner_text()
        self.assertEqual(right_hint, 'CORRECT')
        self.assertEqual(left_hint, 'NOT AD')
        self.page.keyboard.press('Escape')
        self.page.wait_for_timeout(400)

    def test_all_review_buttons_present(self):
        """4 review buttons exist (Ads, Non-Ads, VLM Spastic, Static)."""
        self.page.click('text=Screenshots')
        self.page.wait_for_timeout(800)
        review_btns = self.page.query_selector_all('.review-btn')
        self.assertEqual(len(review_btns), 4, "Should have 4 review buttons")


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright not installed")
class TestReviewModalMobile(unittest.TestCase):
    """Test review modal on mobile viewport (375x812 iPhone X)."""

    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.page = self.browser.new_page(viewport={'width': 375, 'height': 812})
        self.page.goto(BASE_URL)
        self.page.wait_for_timeout(2000)

    def tearDown(self):
        self.page.close()

    def test_tabs_dont_overflow(self):
        """Screenshot tabs and review buttons fit or scroll on mobile."""
        self.page.click('text=Screenshots')
        self.page.wait_for_timeout(800)

        tabs_container = self.page.query_selector('.screenshot-tabs')
        self.assertIsNotNone(tabs_container)

        # Check that container doesn't break layout (no horizontal page scroll)
        page_scroll_width = self.page.evaluate('document.body.scrollWidth')
        viewport_width = self.page.evaluate('window.innerWidth')
        self.assertLessEqual(
            page_scroll_width, viewport_width + 5,
            f"Page should not have horizontal scroll (scroll={page_scroll_width}, viewport={viewport_width})"
        )
        save_screenshot(self.page, "mobile_tabs_fit")

    def test_review_buttons_visible(self):
        """Review buttons are visible/accessible on mobile."""
        self.page.click('text=Screenshots')
        self.page.wait_for_timeout(800)
        review_btns = self.page.query_selector_all('.review-btn')
        self.assertEqual(len(review_btns), 4)
        # First button should be visible
        box = review_btns[0].bounding_box()
        self.assertIsNotNone(box, "First review button should have a bounding box")

    def test_modal_opens_on_mobile(self):
        """Review modal opens correctly on mobile viewport."""
        open_review_modal(self.page, 0)

        has_open = self.page.evaluate(
            'document.getElementById("review-modal").classList.contains("open")'
        )
        self.assertTrue(has_open)
        save_screenshot(self.page, "mobile_modal_open")

    def test_card_fills_mobile_width(self):
        """Review card fills most of the mobile viewport width."""
        open_review_modal(self.page, 0)

        card = self.page.query_selector('.review-card:not(.stack-1):not(.stack-2)')
        self.assertIsNotNone(card)
        box = card.bounding_box()
        viewport_width = 375
        card_width = box['width']
        self.assertGreater(
            card_width, viewport_width * 0.75,
            f"Card should fill >75% of mobile width (card={card_width}, viewport={viewport_width})"
        )

    def test_mobile_swipe_right(self):
        """Touch swipe right works on mobile viewport."""
        open_review_modal(self.page, 0)

        counter_before = self.page.query_selector('#review-counter').inner_text()
        swipe_card(self.page, 'right', 120)
        self.page.wait_for_timeout(600)

        counter_after = self.page.query_selector('#review-counter').inner_text()
        self.assertNotEqual(counter_before, counter_after, "Swipe right should advance on mobile")

    def test_mobile_swipe_left(self):
        """Touch swipe left works on mobile viewport."""
        open_review_modal(self.page, 0)

        counter_before = self.page.query_selector('#review-counter').inner_text()
        swipe_card(self.page, 'left', 120)
        self.page.wait_for_timeout(600)

        counter_after = self.page.query_selector('#review-counter').inner_text()
        self.assertNotEqual(counter_before, counter_after, "Swipe left should advance on mobile")

    def test_hints_visible_on_mobile(self):
        """Action hints are visible at bottom of modal on mobile."""
        open_review_modal(self.page, 0)

        left_hint = self.page.query_selector('.hint-left')
        right_hint = self.page.query_selector('.hint-right')
        self.assertIsNotNone(left_hint)
        self.assertIsNotNone(right_hint)

        # Should be within viewport
        left_box = left_hint.bounding_box()
        right_box = right_hint.bounding_box()
        self.assertLess(left_box['y'], 812, "Left hint should be visible")
        self.assertLess(right_box['y'], 812, "Right hint should be visible")

    def test_undo_on_mobile(self):
        """Undo button works on mobile."""
        open_review_modal(self.page, 0)

        self.page.keyboard.press('ArrowRight')
        self.page.wait_for_timeout(600)

        # The undo server call queues behind the swipe's POST; poll instead
        # of a fixed sleep so a slow POST under load doesn't flake the test.
        self.page.click('.review-undo-btn')
        counter = ''
        for _ in range(20):
            self.page.wait_for_timeout(250)
            counter = self.page.query_selector('#review-counter').inner_text()
            if '1 of' in counter:
                break
        self.assertIn('1 of', counter, "Should be back to first card after undo")


@unittest.skipUnless(HAS_PLAYWRIGHT, "playwright not installed")
class TestReviewAPI(unittest.TestCase):
    """Test the screenshot review API endpoints directly."""

    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch()

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()

    def setUp(self):
        self.page = self.browser.new_page()

    def tearDown(self):
        self.page.close()

    def test_review_endpoint_returns_items(self):
        """GET /api/screenshots/review/ads returns unreviewed items."""
        resp = self.page.goto(f'{BASE_URL}/api/screenshots/review/ads')
        data = resp.json()
        self.assertIn('items', data)
        self.assertIn('total', data)
        self.assertIn('unreviewed', data)
        self.assertIsInstance(data['items'], list)

    def test_review_items_sorted_oldest_first(self):
        """Review items are returned oldest first."""
        resp = self.page.goto(f'{BASE_URL}/api/screenshots/review/ads')
        data = resp.json()
        if len(data['items']) >= 2:
            # Items should be sorted by mtime (oldest first)
            # Filenames contain timestamps, so lexicographic order works
            names = [item['name'] for item in data['items'][:5]]
            self.assertEqual(names, sorted(names), "Items should be sorted oldest first")

    def test_review_invalid_category(self):
        """Invalid category returns 400."""
        resp = self.page.goto(f'{BASE_URL}/api/screenshots/review/invalid')
        self.assertEqual(resp.status, 400)

    def test_approve_and_undo_cycle(self):
        """Approve a screenshot, verify reviewed count changes, then undo."""
        # Get first item
        resp = self.page.goto(f'{BASE_URL}/api/screenshots/review/ads')
        data = resp.json()
        if not data['items']:
            self.skipTest("No ads screenshots to test with")

        first = data['items'][0]['name']
        pre_approve = data['unreviewed']

        # Approve
        resp = self.page.evaluate('''async () => {
            const r = await fetch('/api/screenshots/approve', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({filename: '%s', category: 'ads'})
            });
            return await r.json();
        }''' % first)
        self.assertTrue(resp.get('success'))

        # Check count decreased by 1
        resp2 = self.page.goto(f'{BASE_URL}/api/screenshots/review/ads')
        data2 = resp2.json()
        self.assertEqual(data2['unreviewed'], pre_approve - 1, "Approve should decrease unreviewed by 1")

        # Undo
        resp3 = self.page.evaluate('''async () => {
            const r = await fetch('/api/screenshots/undo', {method: 'POST'});
            return await r.json();
        }''')
        self.assertTrue(resp3.get('success'))

        # Check count increased by 1 from post-approve
        resp4 = self.page.goto(f'{BASE_URL}/api/screenshots/review/ads')
        data4 = resp4.json()
        self.assertEqual(data4['unreviewed'], data2['unreviewed'] + 1, "Undo should increase unreviewed by 1")


if __name__ == '__main__':
    if not HAS_PLAYWRIGHT:
        print("Install playwright: pip3 install playwright && python3 -m playwright install chromium")
        sys.exit(1)

    unittest.main(verbosity=2)
