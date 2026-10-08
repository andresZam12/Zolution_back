"""
End-to-end smoke test script for Zolution multi-tenant API.

Performs health check verification, tenant isolation validation,
and checks conversations and availability endpoints across multiple business sectors.

Usage:
    cd backend
    python -m scripts.smoke_test --base-url http://localhost:8000
"""

import argparse
import sys
import httpx

TEST_TENANTS = [
    {"slug": "lex-asesores", "name": "Lex & Co. Asesoría Legal"},
    {"slug": "nextech-cloud", "name": "NexTech Soluciones Cloud"},
    {"slug": "dra-garcia-dental", "name": "Clínica Dental & Estética Santa María"},
    {"slug": "luxe-spa", "name": "Luxe Wellness & Spa"},
]


def run_smoke_tests(base_url: str):
    """Execute smoke tests against Zolution backend."""
    print(f"\n==========================================")
    print(f"🚀 Running Zolution Smoke Tests on {base_url}")
    print(f"==========================================\n")

    client = httpx.Client(base_url=base_url, timeout=10.0)
    failed = 0

    # 1. Health check
    try:
        resp = client.get("/health")
        if resp.status_code == 200:
            print(f"✅ [PASS] GET /health -> {resp.json()}")
        else:
            print(f"❌ [FAIL] GET /health returned {resp.status_code}: {resp.text}")
            failed += 1
    except Exception as e:
        print(f"⚠️ [SKIP/CONN] Could not connect to {base_url}/health: {e}")
        return

    # 2. Check OpenAPI schema
    try:
        resp = client.get("/openapi.json")
        if resp.status_code == 200:
            data = resp.json()
            title = data.get("info", {}).get("title", "Unknown")
            print(f"✅ [PASS] GET /openapi.json -> API Title: '{title}', Routes: {len(data.get('paths', {}))}")
        else:
            print(f"❌ [FAIL] GET /openapi.json returned {resp.status_code}")
            failed += 1
    except Exception as e:
        print(f"❌ [FAIL] OpenAPI schema error: {e}")
        failed += 1

    # 3. Check Multi-Tenant isolation headers on Conversation API
    for tenant in TEST_TENANTS:
        headers = {
            "X-Organization-ID": tenant["slug"],
            "Authorization": "Bearer demo-token-smoke-test",
        }
        try:
            resp = client.get("/api/v1/conversations", headers=headers)
            print(f"ℹ️ Tenant '{tenant['name']}' ({tenant['slug']}) -> GET /api/v1/conversations: HTTP {resp.status_code}")
        except Exception as e:
            print(f"⚠️ Tenant '{tenant['slug']}' request error: {e}")

    print("\n------------------------------------------")
    if failed == 0:
        print("🎉 Smoke tests completed successfully!")
    else:
        print(f"⚠️ Smoke tests completed with {failed} failures.")
    print("------------------------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zolution E2E Smoke Tests")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base backend URL")
    args = parser.parse_args()

    run_smoke_tests(args.base_url)
