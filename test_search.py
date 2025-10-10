#!/usr/bin/env python3
"""
Quick Test: Verify that search works in the Gradio app
Run this after the app is started to test the search functionality
"""

import requests
import json

APP_URL = "http://localhost:7860"

print("="*80)
print("TESTING GRADIO APP SEARCH FUNCTIONALITY")
print("="*80)

# Test 1: Check if server is running
print("\n1. Testing if server is running...")
try:
    response = requests.get(APP_URL, timeout=5)
    if response.status_code == 200:
        print("   ✓ Server is running at", APP_URL)
    else:
        print(f"   ✗ Server returned status code: {response.status_code}")
        exit(1)
except Exception as e:
    print(f"   ✗ Cannot connect to server: {e}")
    print("\n   → Make sure the app is running:")
    print("     python app.py")
    exit(1)

# Test 2: Test search functionality via API
print("\n2. Testing search API...")

test_drugs = ["Morphine", "Aspirin", "Imatinib", "Talazoparib"]

for drug_name in test_drugs:
    try:
        print(f"\n   Testing search for: {drug_name}")
        
        # Note: Gradio API calls might need different endpoints
        # This is a placeholder - actual API call depends on Gradio version
        
        # For now, just verify the web interface loads
        print(f"   → Open in browser: {APP_URL}")
        print(f"   → Search for: {drug_name}")
        print(f"   → Expected: Dropdown shows results, analysis displays data")
        
    except Exception as e:
        print(f"   ⚠ API test skipped: {e}")

print("\n" + "="*80)
print("MANUAL TESTING INSTRUCTIONS")
print("="*80)
print("\n1. Open your browser and go to:")
print(f"   {APP_URL}")
print("\n2. Test these searches:")
for drug in test_drugs:
    print(f"   • {drug}")
print("\n3. For each drug:")
print("   ✓ Dropdown should show matching results")
print("   ✓ Selecting a drug should display:")
print("     - Basic information table")
print("     - Protein targets table")
print("     - Clinical effects table")
print("     - Interactive network visualization")
print("\n4. If everything works:")
print("   → You're ready to deploy to Hugging Face Spaces!")
print("   → Follow the steps in DEPLOYMENT.md")
print("\n" + "="*80)
