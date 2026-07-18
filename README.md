# Concrete Compressive Strength Predictor

<img width="854" height="954" alt="image" src="https://github.com/user-attachments/assets/11d3a1ff-faf6-4752-8be2-82023251cf28" />


A web-based application that predicts concrete compressive strength using an XGBoost machine learning model, featuring an integrated AI assistant (powered by Gemini 2.5) for concrete materials and mix-design guidance.

**Live application:** https://derrickmirindi.github.io/compressive_strength/

## Authors

**Developed by Derrick Mirindi, David Sinkhonde, Frederic Mirindi and Fatiha Abba.**

## Availability

The web interface is available **24/7 (24 hours a day, 7 days a week)**. The application is hosted on a highly available, always-on infrastructure:

- **Frontend:** Served via GitHub Pages, a globally distributed static hosting service.
- **AI backend:** Powered by a Cloudflare Worker running on Cloudflare's global edge network, ensuring continuous, serverless uptime.

## Security — API Key Protection

The Gemini API key is **never exposed to end users**. All AI requests are routed through a Cloudflare Worker that acts as a secure server-side proxy:

- The API key is stored as a Cloudflare Worker environment secret and is injected into requests **server-side only**.
- The browser communicates only with the Worker endpoint; the key is never included in the client-side code, network responses, or page source.
- This design keeps the credential private and prevents unauthorized use.

## How It Works

1. The user enters concrete mix parameters in the browser.
2. The XGBoost model predicts the compressive strength in-browser.
3. For questions, the AI assistant sends the message to the Cloudflare Worker, which securely calls the Gemini API and returns the response.
