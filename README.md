# 🧭 AI Decision Helper

A Streamlit app that helps you think through a decision. Enter a decision and 2-4 options (plus your priorities) and the app, powered by Google's Gemini Flash model, will:

- list **pros and cons** for each option
- identify **potential risks** (likelihood, impact, mitigation)
- **compare** the options against *your* priorities
- give a **recommendation with step-by-step reasoning**
- separate **facts** from **subjective judgments**, and list assumptions and things to verify

> The recommendation is an AI-generated opinion based only on what you type in. It is not a fact and not professional advice. For medical, legal or financial decisions, consult a qualified professional.

## Project files

```
app.py            # the Streamlit app
requirements.txt  # Python dependencies
README.md         # this file
```

## Run locally

1. Install Python 3.10+.
2. Get a free Gemini API key at <https://aistudio.google.com/apikey>.
3. In the project folder:

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. Provide your key, either way:
   - **Environment variable:** `export GEMINI_API_KEY="your-key"` (Windows PowerShell: `$env:GEMINI_API_KEY="your-key"`)
   - **Secrets file:** create `.streamlit/secrets.toml` containing:

     ```toml
     GEMINI_API_KEY = "your-key"
     ```

   (Or just paste the key into the app's sidebar.)

5. Start the app:

   ```bash
   streamlit run app.py
   ```

### Choosing the model

The default model is `gemini-3.5-flash`. Model names change over time. If you get a "model not found" error, set another Flash model name in the sidebar, or set `GEMINI_MODEL` in your secrets or environment (see the model list in the Gemini API docs).

## Upload the project to GitHub

**Important:** never commit your API key.

1. Create a `.gitignore` file in the project folder with these lines:

   ```
   .venv/
   __pycache__/
   .streamlit/secrets.toml
   ```

2. Create an account at <https://github.com> if you don't have one, then click **New repository**. Name it (e.g. `decision-helper`), choose **Public** (the free Streamlit Community Cloud tier deploys from public repos; private repos need to be connected via your Streamlit account settings), and **don't** tick "Add a README" (you already have one). Click **Create repository**.

3. In a terminal inside your project folder:

   ```bash
   git init
   git add app.py requirements.txt README.md .gitignore
   git commit -m "Initial commit: AI Decision Helper"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/decision-helper.git
   git push -u origin main
   ```

   If asked to sign in, use your GitHub username and a **personal access token** as the password (GitHub → Settings → Developer settings → Personal access tokens), or sign in through GitHub Desktop / the browser prompt.

   *No Git installed?* On the repository page click **Add file → Upload files**, drag in `app.py`, `requirements.txt` and `README.md`, and click **Commit changes**.

## Deploy on Streamlit Community Cloud

1. Go to <https://share.streamlit.io> and sign in with your GitHub account (authorize Streamlit to access your repositories when prompted).
2. Click **Create app** (or **New app**) and choose to deploy from an existing repo.
3. Fill in:
   - **Repository:** `YOUR-USERNAME/decision-helper`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - (Optional) pick a custom app URL.
4. Open **Advanced settings** and, in the **Secrets** box, paste:

   ```toml
   GEMINI_API_KEY = "your-key"
   ```

   (Optionally add `GEMINI_MODEL = "model-name"`.) Click **Save**.
5. Click **Deploy**. The first build installs `requirements.txt` and takes a few minutes. You'll get a public `https://your-app-name.streamlit.app` link.
6. To update the app later, push changes to GitHub (`git add . && git commit -m "update" && git push`). Streamlit redeploys automatically. To change secrets later: app menu → **Settings → Secrets**.

### Troubleshooting

| Problem | Fix |
| --- | --- |
| `ModuleNotFoundError` on deploy | Check that `requirements.txt` is in the repo root and lists `streamlit` and `google-genai`. |
| "API key not valid" / request failed | Re-check the key in Secrets (no extra spaces or quotes inside the key) and reboot the app. |
| "Model not found" | Change the model name in the sidebar or `GEMINI_MODEL` secret. |
| Quota / 429 errors | You've hit the free-tier rate limit. Wait a minute or enable billing on your Google AI project. |
| App is asleep | Free apps sleep after inactivity. Click "Yes, get this app back up!". |

## Notes on responsible use

- The app tells the model to treat your input as data, to label opinions as opinions, and to state its assumptions, but AI output can still be wrong. Verify important facts yourself.
- Your inputs are sent to Google's Gemini API. Don't enter passwords or sensitive personal data.
- If you share a public deployment, your API key is used for everyone's requests. Consider adding usage limits or requiring users to bring their own key (the sidebar field supports this if you leave the secret unset).# AIDecisionMaker
