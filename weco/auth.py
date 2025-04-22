# weco/auth.py
import os
import pathlib
import json
import stat

CONFIG_DIR = pathlib.Path.home() / ".config" / "weco"
CREDENTIALS_FILE = CONFIG_DIR / "credentials.json"


def ensure_config_dir():
    """Ensures the configuration directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Ensure directory permissions are secure (optional but good practice)
    try:
        os.chmod(CONFIG_DIR, stat.S_IRWXU)  # Read/Write/Execute for owner only
    except OSError as e:
        print(f"Warning: Could not set permissions on {CONFIG_DIR}: {e}")


def save_api_key(api_key: str):
    """Saves the Weco API key securely."""
    ensure_config_dir()
    credentials = {"api_key": api_key}
    try:
        with open(CREDENTIALS_FILE, "w") as f:
            json.dump(credentials, f)
        # Set file permissions to read/write for owner only (600)
        os.chmod(CREDENTIALS_FILE, stat.S_IRUSR | stat.S_IWUSR)
    except IOError as e:
        print(f"Error: Could not write credentials file at {CREDENTIALS_FILE}: {e}")
    except OSError as e:
        print(f"Warning: Could not set permissions on {CREDENTIALS_FILE}: {e}")


def load_weco_api_key() -> str | None:
    """Loads the Weco API key."""
    if not CREDENTIALS_FILE.exists():
        return None
    try:
        # Check permissions before reading (optional but safer)
        file_stat = os.stat(CREDENTIALS_FILE)
        if file_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO):  # Check if group/other have permissions
            print(f"Warning: Credentials file {CREDENTIALS_FILE} has insecure permissions. Please set to 600.")
            # Optionally, refuse to load or try to fix permissions

        with open(CREDENTIALS_FILE, "r") as f:
            credentials = json.load(f)
            return credentials.get("api_key")
    except (IOError, json.JSONDecodeError, OSError) as e:
        print(f"Warning: Could not read or parse credentials file at {CREDENTIALS_FILE}: {e}")
        return None


def clear_api_key():
    """Removes the stored API key."""
    if CREDENTIALS_FILE.exists():
        try:
            os.remove(CREDENTIALS_FILE)
            print("Logged out successfully.")
        except OSError as e:
            print(f"Error: Could not remove credentials file at {CREDENTIALS_FILE}: {e}")
    else:
        print("Already logged out.")
