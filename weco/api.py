from typing import Dict, Any, Optional
import rich
import requests
from weco import __pkg_version__, __base_url__
import sys
from rich.console import Console


def handle_api_error(e: requests.exceptions.HTTPError, console: rich.console.Console) -> None:
    """Extract and display error messages from API responses in a structured format."""
    try:
        detail = e.response.json()["detail"]
    except (ValueError, KeyError):  # Handle cases where response is not JSON or detail key is missing
        detail = f"HTTP {e.response.status_code} Error: {e.response.text}"
    console.print(f"[bold red]{detail}[/]")
    # Avoid exiting here, let the caller decide if the error is fatal
    # sys.exit(1)


def start_optimization_session(
    console: rich.console.Console,
    source_code: str,
    evaluation_command: str,
    metric_name: str,
    maximize: bool,
    steps: int,
    code_generator_config: Dict[str, Any],
    evaluator_config: Dict[str, Any],
    search_policy_config: Dict[str, Any],
    additional_instructions: str = None,
    api_keys: Dict[str, Any] = {},
    auth_headers: dict = {},  # Add auth_headers
    timeout: int = 800,
) -> Dict[str, Any]:
    """Start the optimization session."""
    with console.status("[bold green]Starting Optimization..."):
        try:
            response = requests.post(
                f"{__base_url__}/sessions",  # Path is relative to base_url
                json={
                    "source_code": source_code,
                    "additional_instructions": additional_instructions,
                    "objective": {"evaluation_command": evaluation_command, "metric_name": metric_name, "maximize": maximize},
                    "optimizer": {
                        "steps": steps,
                        "code_generator": code_generator_config,
                        "evaluator": evaluator_config,
                        "search_policy": search_policy_config,
                    },
                    "metadata": {"client_name": "cli", "client_version": __pkg_version__, **api_keys},
                },
                headers=auth_headers,  # Add headers
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            handle_api_error(e, console)
            sys.exit(1)  # Exit if starting session fails
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Network Error starting session: {e}[/]")
            sys.exit(1)


def evaluate_feedback_then_suggest_next_solution(
    session_id: str,
    execution_output: str,
    additional_instructions: str = None,
    api_keys: Dict[str, Any] = {},
    auth_headers: dict = {},  # Add auth_headers
    timeout: int = 800,
) -> Dict[str, Any]:
    """Evaluate the feedback and suggest the next solution."""
    try:
        response = requests.post(
            f"{__base_url__}/sessions/{session_id}/suggest",  # Path is relative to base_url
            json={
                "execution_output": execution_output,
                "additional_instructions": additional_instructions,
                "metadata": {**api_keys},
            },
            headers=auth_headers,  # Add headers
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        # Allow caller to handle suggest errors, maybe retry or terminate
        handle_api_error(e, Console())  # Use default console if none passed
        raise  # Re-raise the exception
    except requests.exceptions.RequestException as e:
        print(f"Network Error during suggest: {e}")  # Use print as console might not be available
        raise  # Re-raise the exception


def get_optimization_session_status(
    session_id: str, include_history: bool = False, auth_headers: dict = {}, timeout: int = 800
) -> Dict[str, Any]:
    """Get the current status of the optimization session."""
    try:
        response = requests.get(
            f"{__base_url__}/sessions/{session_id}",  # Path is relative to base_url
            params={"include_history": include_history},
            headers=auth_headers,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        handle_api_error(e, Console())  # Use default console
        raise  # Re-raise
    except requests.exceptions.RequestException as e:
        print(f"Network Error getting status: {e}")
        raise  # Re-raise


def send_heartbeat(
    session_id: str,
    auth_headers: dict = {},
    timeout: int = 10,  # Shorter timeout for non-critical heartbeat
) -> bool:
    """Send a heartbeat signal to the backend."""
    try:
        response = requests.put(f"{__base_url__}/sessions/{session_id}/heartbeat", headers=auth_headers, timeout=timeout)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        return True
    except requests.exceptions.HTTPError as e:
        # Log non-critical errors like 409 Conflict (session not running)
        if e.response.status_code == 409:
            print(f"Heartbeat ignored: Session {session_id} is not running.", file=sys.stderr)
        else:
            print(f"Heartbeat failed for session {session_id}: HTTP {e.response.status_code}", file=sys.stderr)
        # Don't exit, just report failure
        return False
    except requests.exceptions.RequestException as e:
        # Network errors are also non-fatal for heartbeats
        print(f"Heartbeat network error for session {session_id}: {e}", file=sys.stderr)
        return False


def report_termination(
    session_id: str,
    status_update: str,
    reason: str,
    details: Optional[str] = None,
    auth_headers: dict = {},
    timeout: int = 30,  # Reasonably longer timeout for important termination message
) -> bool:
    """Report the termination reason to the backend."""
    try:
        response = requests.post(
            f"{__base_url__}/sessions/{session_id}/terminate",
            json={"status_update": status_update, "termination_reason": reason, "termination_details": details},
            headers=auth_headers,
            timeout=timeout,
        )
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        # Log failure, but don't prevent CLI exit
        print(f"Warning: Failed to report termination to backend for session {session_id}: {e}", file=sys.stderr)
        return False
