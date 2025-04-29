from typing import Dict, Any
import rich
import requests
from weco import __pkg_version__, __base_url__
import sys


def handle_api_error(e: requests.exceptions.HTTPError, console: rich.console.Console) -> None:
    """Extract and display error messages from API responses in a structured format."""
    console.print(f"[bold red]{e.response.json()['detail']}[/]")
    sys.exit(1)


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


def evaluate_feedback_then_suggest_next_solution(
    session_id: str,
    execution_output: str,
    additional_instructions: str = None,
    api_keys: Dict[str, Any] = {},
    auth_headers: dict = {},  # Add auth_headers
    timeout: int = 800,
) -> Dict[str, Any]:
    """Evaluate the feedback and suggest the next solution."""
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


def get_optimization_session_status(
    session_id: str, include_history: bool = False, auth_headers: dict = {}, timeout: int = 800
) -> Dict[str, Any]:
    """Get the current status of the optimization session."""
    response = requests.get(
        f"{__base_url__}/sessions/{session_id}",  # Path is relative to base_url
        params={"include_history": include_history},
        headers=auth_headers,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()
