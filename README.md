# OnlyFunds

## Getting Started

1. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
2. **Configure**
   - Edit `config/config.yaml` for all trading, risk, strategy, and API settings.
   - Secrets can be set as environment variables for security.

3. **Run the App**
   - Double-click `START.bat` (Windows)
   - Or run:
     ```
     streamlit run main.py
     ```

## Configuration

- **All settings are in** `config/config.yaml`.
- Use the sidebar to override key settings during a session.
- Click **💾 Save Preferences** in the sidebar to persist your changes back to `config/config.yaml`.
- Click **🔃 Reset Sidebar to Config Defaults** to reload the sidebar from your config.

## Prometheus Monitoring

- Prometheus config is in `config/prometheus.yml`.
- Default scrape target: `localhost:8000`
- Start the Prometheus server with this config:
  ```
  prometheus --config.file=config/prometheus.yml
  ```

## Repo Hygiene

- Runtime state/logs are ignored by `.gitignore` (`state/`, `logs/`).
- Only `config/config.yaml` is used for configuration (no JSON configs).
- Clean up obsolete files from `config/`.

## Features

- Streamlit sidebar for live parameter changes.
- "Save Preferences" button for persistent config.
- Robust, production-grade structure.

---

**Happy trading!**