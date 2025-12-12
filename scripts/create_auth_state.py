#!/usr/bin/env python3
"""
Create Playwright storage state for WebArena authentication.
This allows BrowserGym to start with authenticated sessions.
"""
import os
import asyncio
from playwright.async_api import async_playwright
from pathlib import Path

# Credentials from .env
USERNAME = os.getenv("SHOPPING_ADMIN_USERNAME", "admin")
PASSWORD = os.getenv("SHOPPING_ADMIN_PASSWORD", "admin1234")
ADMIN_URL = os.getenv("SHOPPING_ADMIN", "http://ec2-3-150-230-27.us-east-2.compute.amazonaws.com:7780/admin")

async def create_storage_state():
    """Create authenticated storage state for Magento admin."""
    
    # Create cache directory
    cache_dir = Path.home() / ".cache" / "browsergym" / "webarena"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    storage_path = cache_dir / "shopping_admin_storage.json"
    
    print(f"Creating authenticated storage state...")
    print(f"URL: {ADMIN_URL}")
    print(f"Username: {USERNAME}")
    print(f"Output: {storage_path}")
    print()
    
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            # Navigate to login page
            print("1. Navigating to admin login...")
            await page.goto(ADMIN_URL, wait_until="domcontentloaded", timeout=60000)
            await page.wait_for_timeout(2000)
            
            # Check if already logged in
            if "/dashboard" in page.url:
                print("   Already logged in!")
            else:
                # Fill login form
                print("2. Filling login credentials...")
                await page.wait_for_selector('input[name="login[username]"]', timeout=10000)
                await page.fill('input[name="login[username]"]', USERNAME)
                await page.fill('input[name="login[password]"]', PASSWORD)
                
                # Click sign in
                print("3. Submitting login...")
                await page.click('button:has-text("Sign in")')
                
                # Wait for URL change (more reliable than specific URL)
                print("4. Waiting for login redirect...")
                await page.wait_for_load_state("networkidle", timeout=60000)
                await page.wait_for_timeout(3000)
            
            # Check final URL
            print(f"   Final URL: {page.url}")
            
            # Verify we're logged in (look for admin elements)
            if "dashboard" in page.url or "admin" in page.url:
                # Save storage state
                print("5. Saving authentication state...")
                await context.storage_state(path=str(storage_path))
                
                print(f"\n✅ Success! Storage state saved to:")
                print(f"   {storage_path}")
                print()
                print("BrowserGym will now use this authenticated session.")
            else:
                print(f"\n⚠️  Warning: May not be fully logged in")
                print(f"   Current URL: {page.url}")
                # Save anyway in case it's valid
                await context.storage_state(path=str(storage_path))
                print(f"   Saved storage state anyway.")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("\nPlease verify:")
            print(f"1. Admin URL is accessible: {ADMIN_URL}")
            print(f"2. Credentials are correct: {USERNAME} / {PASSWORD}")
            raise
        
        finally:
            await browser.close()

if __name__ == "__main__":
    # Load .env if available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Warning: python-dotenv not installed, using existing env vars")
    
    asyncio.run(create_storage_state())
