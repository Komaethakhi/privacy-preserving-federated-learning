import os
import time
from playwright.sync_api import sync_playwright

def run():
    artifacts_dir = r"C:\Users\komae\.gemini\antigravity\brain\025979b5-b351-4b4e-8d70-40e2176c9599"
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs("scratch", exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        print("Navigating to login page...")
        page.goto("http://localhost:8501")
        page.wait_for_selector(".stApp", timeout=15000)
        time.sleep(2)

        # 1. Login Light
        print("Capturing login_light.png...")
        page.screenshot(path="scratch/login_light.png")
        page.screenshot(path=os.path.join(artifacts_dir, "login_light.png"))

        # Toggle to Dark Mode
        print("Toggling to dark mode on login page...")
        page.click("button:has-text('Dark Mode')")
        time.sleep(2)

        # 2. Login Dark
        print("Capturing login_dark.png...")
        page.screenshot(path="scratch/login_dark.png")
        page.screenshot(path=os.path.join(artifacts_dir, "login_dark.png"))

        # Switch back to Light Mode for hospital login
        page.click("button:has-text('Light Mode')")
        time.sleep(1)

        # Login as hospital_a
        print("Logging in as hospital_a...")
        inputs = page.query_selector_all("input[type='text'], input[type='password']")
        if len(inputs) >= 2:
            inputs[0].fill("hospital_a")
            inputs[1].fill("hospital123")
            page.click("button:has-text('LOGIN')")
            time.sleep(3)

        # 3. Hospital Dashboard Light
        print("Capturing hospital_dashboard_light.png...")
        page.screenshot(path="scratch/hospital_dashboard_light.png")
        page.screenshot(path=os.path.join(artifacts_dir, "hospital_dashboard_light.png"))

        # Toggle to Dark Mode
        print("Toggling to dark mode on hospital dashboard...")
        page.click("button:has-text('Dark Mode')")
        time.sleep(2)

        # 4. Hospital Dashboard Dark
        print("Capturing hospital_dashboard_dark.png...")
        page.screenshot(path="scratch/hospital_dashboard_dark.png")
        page.screenshot(path=os.path.join(artifacts_dir, "hospital_dashboard_dark.png"))

        # Logout
        print("Logging out...")
        page.click("button:has-text('Logout')")
        time.sleep(2)

        # Login as admin
        print("Logging in as admin...")
        inputs = page.query_selector_all("input[type='text'], input[type='password']")
        if len(inputs) >= 2:
            inputs[0].fill("admin")
            inputs[1].fill("admin123")
            page.click("button:has-text('LOGIN')")
            time.sleep(3)

        # Admin currently in Dark or Light depending on state. Check button text
        toggle_btn = page.query_selector("button:has-text('Dark Mode')")
        if toggle_btn:
            # Currently light, screenshot light first
            print("Capturing admin_overview_light.png...")
            page.screenshot(path="scratch/admin_overview_light.png")
            page.screenshot(path=os.path.join(artifacts_dir, "admin_overview_light.png"))

            print("Toggling to dark mode on admin overview...")
            toggle_btn.click()
            time.sleep(2)

            print("Capturing admin_overview_dark.png...")
            page.screenshot(path="scratch/admin_overview_dark.png")
            page.screenshot(path=os.path.join(artifacts_dir, "admin_overview_dark.png"))
        else:
            # Currently dark
            print("Capturing admin_overview_dark.png...")
            page.screenshot(path="scratch/admin_overview_dark.png")
            page.screenshot(path=os.path.join(artifacts_dir, "admin_overview_dark.png"))

            print("Toggling to light mode on admin overview...")
            page.click("button:has-text('Light Mode')")
            time.sleep(2)

            print("Capturing admin_overview_light.png...")
            page.screenshot(path="scratch/admin_overview_light.png")
            page.screenshot(path=os.path.join(artifacts_dir, "admin_overview_light.png"))

        browser.close()
        print("All 6 screenshots captured successfully.")

if __name__ == "__main__":
    run()
