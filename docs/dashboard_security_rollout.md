# Dashboard authentication rollout

This change deliberately fails closed. The Render health probe at /health stays public, but returns 503 until the auth configuration is valid. Every dashboard page, static asset, and API route requires server-side HTTP Basic Authentication. Unsafe methods (POST, PUT, PATCH, DELETE) also require an exact same-origin Origin or Referer.

## Configure Render before deploying

In the cryptobot-dashboard Render service, add these environment variables through the Render dashboard (not source code):

- DASHBOARD_AUTH_USERNAME: a non-default admin username.
- DASHBOARD_AUTH_PASSWORD: a unique password of at least 20 characters; use a password manager to generate and store it.
- DASHBOARD_PUBLIC_ORIGIN: the exact HTTPS origin users open, e.g. https://your-actual-service-domain.onrender.com (no path; use the actual Render/custom domain).

Do not put credentials in dashboard_static/index.html, GitHub Actions workflow YAML values, source control, logs, issue comments, or chat transcripts. The password must not be reused as a GitHub PAT or exchange credential.

If any variable is missing, the password is too short, or the origin is invalid, private routes return HTTP 503 and /health also returns 503 rather than showing the service as ready. Once configured, /health returns 200 and only status/time. Configure all three values before deploying, and verify them without printing their values.

## Verification after deployment

1. GET /health returns 200 only after authentication config is valid, and contains only status/time.
2. GET / and GET /api/status without credentials return 401 with a Basic Auth challenge.
3. Browser login with the configured credentials loads the dashboard and its same-origin API calls.
4. An authenticated cross-origin POST is rejected with 403; requests with neither Origin nor same-origin Referer are rejected.
5. Missing/weak configuration gives 503 for private endpoints; never add a plaintext/public fallback.

HTTP Basic Auth is an interim access-control layer; TLS must be provided by the public hosting platform. This patch does not implement runtime-state encryption or remove historical plaintext state from Git. Those remain separate security tasks and must not be considered resolved by dashboard authentication.
