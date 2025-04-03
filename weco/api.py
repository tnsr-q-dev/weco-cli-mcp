from typing import Dict, Any
import rich
import requests
from weco import __pkg_version__, __base_url__
import sys


def handle_api_error(e: requests.exceptions.HTTPError, console: rich.console.Console) -> None:
    """Extract and display error messages from API responses."""
    try:
        error_data = e.response.json()
        error_message = error_data.get("detail", str(e))
        console.print(f"[bold red]Server Error:[/] {error_message}")
    except Exception:
        # If we can't parse the JSON, just show the original error
        console.print(f"[bold red]Server Error:[/] {str(e)}")
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
) -> Dict[str, Any]:
    """Start the optimization session."""
    with console.status("[bold green]Starting Optimization..."):
        try:
            response = requests.post(
                f"{__base_url__}/sessions",
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
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            handle_api_error(e=e, console=console)


def evaluate_feedback_then_suggest_next_solution(
    console: rich.console.Console,
    session_id: str,
    execution_output: str,
    additional_instructions: str = None,
    api_keys: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """Evaluate the feedback and suggest the next solution."""
    try:
        response = requests.post(
            f"{__base_url__}/sessions/{session_id}/suggest",
            json={
                "execution_output": execution_output,
                "additional_instructions": additional_instructions,
                "metadata": {**api_keys},
            },
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        handle_api_error(e=e, console=console)


def get_optimization_session_status(
    console: rich.console.Console, session_id: str, include_history: bool = False
) -> Dict[str, Any]:
    """Get the current status of the optimization session."""
    try:
        response = requests.get(f"{__base_url__}/sessions/{session_id}", params={"include_history": include_history})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        handle_api_error(e=e, console=console)
