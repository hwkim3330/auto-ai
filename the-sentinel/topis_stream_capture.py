#!/usr/bin/env python3
"""
TOPIS CCTV Stream Capture Tool
================================

Bypass 5-second limitation and capture real TOPIS CCTV stream URLs

Method: Selenium automation + Network traffic capture
"""

import time
import json
from typing import List, Dict, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import threading
import queue


class TOPISStreamCapture:
    """Capture TOPIS CCTV stream URLs automatically"""

    def __init__(self, headless: bool = False):
        self.headless = headless
        self.driver = None
        self.captured_streams = {}
        self.network_logs = []

    def setup_driver(self):
        """Setup Selenium WebDriver with network logging"""
        chrome_options = Options()

        if self.headless:
            chrome_options.add_argument('--headless')

        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')

        # Enable network logging
        capabilities = DesiredCapabilities.CHROME
        capabilities['goog:loggingPrefs'] = {'performance': 'ALL'}

        self.driver = webdriver.Chrome(options=chrome_options)
        print("[Driver] Chrome WebDriver initialized")

    def parse_network_log(self) -> List[Dict]:
        """Parse network log to find stream URLs"""
        logs = self.driver.get_log('performance')

        stream_requests = []
        for entry in logs:
            try:
                log = json.loads(entry['message'])['message']

                # Look for network requests
                if log['method'] == 'Network.requestWillBeSent':
                    url = log['params']['request']['url']

                    # Filter for video/stream URLs
                    if any(ext in url for ext in ['.m3u8', '.mp4', '.flv', 'stream', 'video']):
                        stream_requests.append({
                            'url': url,
                            'method': log['params']['request']['method'],
                            'headers': log['params']['request']['headers']
                        })
                        print(f"[Found] Stream URL: {url}")

            except Exception as e:
                continue

        return stream_requests

    def click_cctv_on_map(self, index: int = 0):
        """
        Click CCTV marker on the map

        Note: This requires knowing the actual DOM structure
        You may need to inspect the page and adjust selectors
        """
        try:
            # Wait for map to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "map"))
            )

            # Find CCTV markers (adjust selector based on actual page)
            # Common patterns:
            # - <img> tags with specific class
            # - <div> markers
            # - SVG elements

            # Try common selectors
            selectors = [
                "img.cctv-marker",
                "div.marker-icon",
                ".leaflet-marker-icon",
                "[data-cctv-id]"
            ]

            for selector in selectors:
                try:
                    markers = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if markers and len(markers) > index:
                        print(f"[Click] CCTV marker #{index}")
                        markers[index].click()
                        time.sleep(1)
                        return True
                except Exception as e:
                    continue

            print("[Warning] Could not find CCTV markers automatically")
            print("[Info] You may need to click manually or inspect the page structure")
            return False

        except Exception as e:
            print(f"[Error] {e}")
            return False

    def capture_stream_url(self, duration: int = 10):
        """
        Capture stream URL by monitoring network traffic

        Args:
            duration: How long to monitor (seconds)
        """
        print(f"[Monitoring] Network traffic for {duration} seconds...")

        start_time = time.time()
        all_streams = []

        while time.time() - start_time < duration:
            # Parse current network logs
            streams = self.parse_network_log()
            all_streams.extend(streams)

            time.sleep(1)

        # Deduplicate
        unique_streams = {s['url']: s for s in all_streams}

        print(f"[Captured] {len(unique_streams)} unique stream URLs")
        return list(unique_streams.values())

    def auto_capture_all_cctvs(self, max_cctvs: int = 10):
        """
        Automatically capture stream URLs for multiple CCTVs
        """
        print(f"[Auto-Capture] Starting for up to {max_cctvs} CCTVs")

        for i in range(max_cctvs):
            print(f"\n[CCTV {i+1}/{max_cctvs}]")

            # Click CCTV
            if self.click_cctv_on_map(i):
                # Capture stream
                streams = self.capture_stream_url(duration=6)

                if streams:
                    self.captured_streams[f"CCTV_{i+1}"] = streams[0]
                    print(f"[Success] Captured stream for CCTV_{i+1}")
                else:
                    print(f"[Failed] No stream found for CCTV_{i+1}")

            # Wait before next
            time.sleep(2)

        return self.captured_streams

    def run(self, url: str = "https://topis.seoul.go.kr/map/openCctvMap.do"):
        """
        Main execution flow
        """
        try:
            # Setup
            self.setup_driver()

            # Navigate to TOPIS
            print(f"[Navigate] {url}")
            self.driver.get(url)

            # Wait for page load
            time.sleep(5)

            print("\n" + "=" * 70)
            print("TOPIS STREAM CAPTURE - Ready")
            print("=" * 70)
            print("\nOptions:")
            print("1. Auto-capture (attempt to find streams automatically)")
            print("2. Manual mode (you click, I capture)")
            print("3. Quit")

            choice = input("\nSelect option (1/2/3): ").strip()

            if choice == '1':
                # Auto-capture
                streams = self.auto_capture_all_cctvs(max_cctvs=5)
                self.save_results(streams)

            elif choice == '2':
                # Manual mode
                print("\n[Manual Mode]")
                print("Click any CCTV on the map in the browser window")
                print("I'll capture the stream URL in the background")
                print("Press Ctrl+C when done")

                try:
                    streams = self.capture_stream_url(duration=60)
                    self.save_results({'manual_capture': streams})
                except KeyboardInterrupt:
                    print("\n[Stopped] Manual capture")
                    streams = self.parse_network_log()
                    self.save_results({'manual_capture': streams})

            else:
                print("[Exit]")

        finally:
            if self.driver:
                self.driver.quit()

    def save_results(self, streams: Dict):
        """Save captured streams to JSON"""
        output_file = 'topis_streams.json'

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(streams, f, indent=2, ensure_ascii=False)

        print(f"\n[Saved] Stream URLs saved to {output_file}")

        # Print summary
        print("\n" + "=" * 70)
        print("CAPTURE SUMMARY")
        print("=" * 70)

        for cctv_id, stream_info in streams.items():
            if isinstance(stream_info, dict):
                print(f"\n{cctv_id}:")
                print(f"  URL: {stream_info.get('url', 'N/A')}")
                print(f"  Method: {stream_info.get('method', 'N/A')}")
            elif isinstance(stream_info, list):
                print(f"\n{cctv_id}: {len(stream_info)} URLs captured")
                for idx, s in enumerate(stream_info[:3]):  # Show first 3
                    print(f"  {idx+1}. {s.get('url', 'N/A')}")


def main():
    print("=" * 70)
    print("TOPIS CCTV STREAM CAPTURE TOOL")
    print("서울시 TOPIS CCTV 스트림 URL 추출 도구")
    print("=" * 70)
    print("\nNote: Requires Chrome/Chromium browser")
    print("Install: pip install selenium")
    print()

    try:
        capture = TOPISStreamCapture(headless=False)
        capture.run()

    except Exception as e:
        print(f"\n[Error] {e}")
        print("\nTroubleshooting:")
        print("1. Install ChromeDriver: sudo apt install chromium-chromedriver")
        print("2. Or download from: https://chromedriver.chromium.org/")
        print("3. Make sure Chrome browser is installed")
        print("\nAlternative: Use manual method (see below)")


if __name__ == "__main__":
    main()
