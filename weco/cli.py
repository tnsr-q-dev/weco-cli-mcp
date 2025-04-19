import argparse
import sys
import pathlib
import math
import time  # Add time import
import requests  # Add requests import
import webbrowser  # Add webbrowser import
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.traceback import install
from rich.prompt import Prompt  # For user input
from .api import (  # Keep existing imports
    start_optimization_session,
    evaluate_feedback_then_suggest_next_solution,
    get_optimization_session_status,
    handle_api_error,  # Ensure handle_api_error is imported if used
)
from .auth import load_api_key, save_api_key, clear_api_key  # Import auth functions
from weco import __base_url__  # Import base_url for constructing API paths
from .panels import (
    SummaryPanel,
    PlanPanel,
    Node,
    MetricTreePanel,
    EvaluationOutputPanel,
    SolutionPanels,
    create_optimization_layout,
    create_end_optimization_layout,
)
from .utils import (
    read_api_keys_from_env,
    read_additional_instructions,
    read_from_path,
    write_to_path,
    run_evaluation,
    smooth_update,
    format_number,
)

install(show_locals=True)
console = Console()


# --- New Login Function ---
def perform_login(console: Console):
    """Handles the device login flow."""
    try:
        # 1. Initiate device login
        console.print("Initiating login...")
        # __base_url__ already contains /v1
        init_response = requests.post(f"{__base_url__}/auth/device/initiate")  # REMOVED /v1 prefix
        init_response.raise_for_status()
        init_data = init_response.json()

        device_code = init_data["device_code"]
        verification_uri = init_data["verification_uri"]
        # user_code = init_data.get("user_code") # If using user codes
        expires_in = init_data["expires_in"]
        interval = init_data["interval"]

        # 2. Display instructions
        console.print("\n[bold yellow]Action Required:[/]")
        console.print("Please open the following URL in your browser to authenticate:")
        console.print(f"[link={verification_uri}]{verification_uri}[/link]")
        # if user_code:
        #     console.print(f"And enter the code: [bold cyan]{user_code}[/]")
        console.print(f"This request will expire in {expires_in // 60} minutes.")
        console.print("Attempting to open the authentication page in your default browser...")  # Notify user

        # Automatically open the browser
        try:
            if not webbrowser.open(verification_uri):
                console.print("[yellow]Could not automatically open the browser. Please open the link manually.[/]")
        except Exception as browser_err:
            console.print(
                f"[yellow]Could not automatically open the browser ({browser_err}). Please open the link manually.[/]"
            )

        console.print("Waiting for authentication...", end="")

        # 3. Poll for token
        start_time = time.time()
        # Use a simple text update instead of Spinner within Live for potentially better compatibility
        polling_status = "Waiting..."
        with Live(polling_status, refresh_per_second=1, transient=True, console=console) as live_status:
            while True:
                # Check for timeout
                if time.time() - start_time > expires_in:
                    console.print("\n[bold red]Error:[/] Login request timed out.")
                    return False

                time.sleep(interval)
                live_status.update("Waiting... (checking status)")

                try:
                    token_response = requests.post(
                        f"{__base_url__}/auth/device/token",  # REMOVED /v1 prefix
                        json={"grant_type": "urn:ietf:params:oauth:grant-type:device_code", "device_code": device_code},
                    )

                    # Check for 202 Accepted - Authorization Pending
                    if token_response.status_code == 202:
                        token_data = token_response.json()
                        if token_data.get("error") == "authorization_pending":
                             live_status.update("Waiting... (authorization pending)")
                             continue # Continue polling
                        else:
                            # Unexpected 202 response format
                            console.print(f"\n[bold red]Error:[/] Received unexpected 202 response: {token_data}")
                            return False

                    # Check for standard OAuth2 errors (often 400 Bad Request)
                    elif token_response.status_code == 400:
                        token_data = token_response.json()
                        error_code = token_data.get("error", "unknown_error")
                        # NOTE: Removed "authorization_pending" check from here
                        if error_code == "slow_down":
                            interval += 5  # Increase polling interval if instructed
                            live_status.update(f"Waiting... (slowing down polling to {interval}s)")
                            continue
                        elif error_code == "expired_token":
                            console.print("\n[bold red]Error:[/] Login request expired.")
                            return False
                        elif error_code == "access_denied":
                            console.print("\n[bold red]Error:[/] Authorization denied by user.")
                            return False
                        else:  # invalid_grant, etc.
                            error_desc = token_data.get("error_description", "Unknown error during polling.")
                            console.print(f"\n[bold red]Error:[/] {error_desc} ({error_code})")
                            return False

                    # Check for other non-200/non-202/non-400 HTTP errors
                    token_response.raise_for_status()

                    # If successful (200 OK and no 'error' field)
                    token_data = token_response.json()
                    if "access_token" in token_data:
                        api_key = token_data["access_token"]
                        save_api_key(api_key)
                        console.print("\n[bold green]Login successful![/]")
                        return True
                    else:
                        # Unexpected successful response format
                        console.print("\n[bold red]Error:[/] Received unexpected response from server during polling.")
                        print(token_data)  # Log for debugging
                        return False

                except requests.exceptions.RequestException as e:
                    # Handle network errors during polling gracefully
                    live_status.update("Waiting... (network error, retrying)")
                    console.print(f"\n[bold yellow]Warning:[/] Network error during polling: {e}. Retrying...")
                    # Optional: implement backoff strategy
                    time.sleep(interval * 2)  # Simple backoff

    except requests.exceptions.HTTPError as e:  # Catch HTTPError specifically for handle_api_error
        handle_api_error(e, console)
    except requests.exceptions.RequestException as e:  # Catch other request errors
        console.print(f"\n[bold red]Network Error:[/] {e}")
        return False
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred during login:[/] {e}")
        return False


def main() -> None:
    """Main function for the Weco CLI."""
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="[bold cyan]Weco CLI[/]", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Add subparsers for commands like 'run' and 'logout'
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)  # Make command required

    # --- Run Command ---
    run_parser = subparsers.add_parser(
        "run", help="Run code optimization", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Add arguments specific to the 'run' command to the run_parser
    run_parser.add_argument("--source", type=str, required=True, help="Path to the Python source code (e.g. optimize.py)")
    run_parser.add_argument(
        "--eval-command", type=str, required=True, help="Command to run for evaluation (e.g. 'python eval.py --arg1=val1')"
    )
    run_parser.add_argument("--metric", type=str, required=True, help="Metric to optimize")
    run_parser.add_argument(
        "--maximize",
        type=str,
        choices=["true", "false"],
        required=True,
        help="Specify 'true' to maximize the metric or 'false' to minimize.",
    )
    run_parser.add_argument("--steps", type=int, required=True, help="Number of steps to run")
    run_parser.add_argument("--model", type=str, required=True, help="Model to use for optimization")
    run_parser.add_argument(
        "--additional-instructions",
        default=None,
        type=str,
        help="Description of additional instruction or path to a file containing additional instructions",
    )

    # ADD THE LOGOUT PARSER DEFINITION HERE
    logout_parser = subparsers.add_parser("logout", help="Log out from Weco and clear saved API key.")  # noqa F841

    args = parser.parse_args()

    # --- Handle Logout Command ---
    if args.command == "logout":
        clear_api_key()
        sys.exit(0)

    # --- Handle Run Command ---
    elif args.command == "run":
        # --- Check Authentication ---
        weco_api_key = load_api_key()
        llm_api_keys = read_api_keys_from_env()  # Read keys from client environment

        if not weco_api_key:
            # console.print("You are not logged in.") # Removing redundant message
            login_choice = Prompt.ask(
                # Updated prompt text and choices
                "Log in to Weco to save run history or use anonymously? ([bold]L[/]ogin / [bold]S[/]kip)",
                choices=["l", "s"],  # Only 'l' or 's' are valid inputs now
                default="s",  # Default to skip/anonymous
            ).lower()

            if login_choice == "l":  # User chose to log in
                console.print("[cyan]Starting login process...[/]")
                if not perform_login(console):
                    console.print("[bold red]Login process failed or was cancelled.[/]")
                    sys.exit(1)  # Exit if login failed
                weco_api_key = load_api_key()  # Reload key after successful login
                if not weco_api_key:  # Check if key was actually saved
                    console.print("[bold red]Error: Login completed but failed to retrieve API key.[/]")
                    sys.exit(1)
                console.print("[green]Login successful. Proceeding with authenticated run.[/]")

            elif login_choice == "s":  # User chose to skip
                console.print("[yellow]Proceeding anonymously. LLM API keys must be provided via environment variables.[/]")
                if not llm_api_keys:
                    console.print(
                        "[bold red]Error:[/] No LLM API keys found in environment (e.g., OPENAI_API_KEY). Cannot proceed anonymously."
                    )
                    sys.exit(1)
            # No else needed as choices are restricted

        # --- Prepare API Call Arguments ---
        auth_headers = {}
        # Always pass LLM keys found in client env to metadata
        metadata_keys = llm_api_keys.copy()

        if weco_api_key:
            auth_headers["Authorization"] = f"Bearer {weco_api_key}"
            # Don't clear metadata_keys here, backend will decide based on auth status

        # --- Main Run Logic ---
        try:
            # --- Configuration Loading ---
            evaluation_command = args.eval_command
            metric_name = args.metric
            maximize = args.maximize == "true"
            steps = args.steps
            code_generator_config = {"model": args.model}
            evaluator_config = {"model": args.model}
            search_policy_config = {
                "num_drafts": max(1, math.ceil(0.15 * steps)),
                "debug_prob": 0.5,
                "max_debug_depth": max(1, math.ceil(0.1 * steps)),
            }
            additional_instructions = read_additional_instructions(additional_instructions=args.additional_instructions)
            source_fp = pathlib.Path(args.source)
            source_code = read_from_path(fp=source_fp, is_json=False)
            timeout = 800

            # --- Panel Initialization ---
            summary_panel = SummaryPanel(maximize=maximize, metric_name=metric_name, total_steps=steps, model=args.model)
            plan_panel = PlanPanel()
            solution_panels = SolutionPanels(metric_name=metric_name)
            eval_output_panel = EvaluationOutputPanel()
            tree_panel = MetricTreePanel(maximize=maximize)
            layout = create_optimization_layout()
            end_optimization_layout = create_end_optimization_layout()

            # --- Start Optimization Session ---
            console.print("[cyan]Starting optimization session...[/]")
            session_response = start_optimization_session(
                console=console,
                source_code=source_code,
                evaluation_command=evaluation_command,
                metric_name=metric_name,
                maximize=maximize,
                steps=steps,
                code_generator_config=code_generator_config,
                evaluator_config=evaluator_config,
                search_policy_config=search_policy_config,
                additional_instructions=additional_instructions,
                api_keys=metadata_keys,  # Pass client LLM keys
                auth_headers=auth_headers,  # Pass Weco key if logged in
                timeout=timeout,
            )

            # --- Live Update Loop ---
            refresh_rate = 4
            with Live(layout, refresh_per_second=refresh_rate, screen=True) as live:
                session_id = session_response["session_id"]
                runs_dir = pathlib.Path(".runs") / session_id
                runs_dir.mkdir(parents=True, exist_ok=True)

                runs_copy_source_fp = runs_dir / "original.py"
                write_to_path(fp=runs_copy_source_fp, content=source_code)
                write_to_path(fp=source_fp, content=session_response["code"])

                summary_panel.session_id = session_id
                summary_panel.set_step(step=0)
                summary_panel.update_token_counts(usage=session_response["usage"])
                plan_panel.update(plan=session_response["plan"])
                tree_panel.build_metric_tree(
                    nodes=[
                        {
                            "solution_id": session_response["solution_id"],
                            "parent_id": None,
                            "code": session_response["code"],
                            "step": 0,
                            "metric_value": None,
                            "is_buggy": False,
                        }
                    ]
                )
                tree_panel.set_unevaluated_node(node_id=session_response["solution_id"])
                solution_panels.update(
                    current_node=Node(
                        id=session_response["solution_id"],
                        parent_id=None,
                        code=session_response["code"],
                        metric=None,
                        is_buggy=False,
                    ),
                    best_node=None,
                )
                current_solution_panel, best_solution_panel = solution_panels.get_display(current_step=0)

                smooth_update(
                    live=live,
                    layout=layout,
                    sections_to_update=[
                        ("summary", summary_panel.get_display()),
                        ("plan", plan_panel.get_display()),
                        ("tree", tree_panel.get_display()),
                        ("current_solution", current_solution_panel),
                        ("best_solution", best_solution_panel),
                        ("eval_output", eval_output_panel.get_display()),
                    ],
                    transition_delay=0.1,
                )

                term_out = run_evaluation(eval_command=args.eval_command)
                eval_output_panel.update(output=term_out)
                smooth_update(
                    live=live,
                    layout=layout,
                    sections_to_update=[("eval_output", eval_output_panel.get_display())],
                    transition_delay=0.1,
                )

                best_solution_node = None
                for step in range(steps):
                    if step > 0:
                        term_out = run_evaluation(eval_command=args.eval_command)
                        eval_output_panel.update(output=term_out)
                        smooth_update(live, layout, [("eval_output", eval_output_panel.get_display())])

                    is_last_evaluation = step == steps - 1

                    eval_and_next_solution_response = evaluate_feedback_then_suggest_next_solution(
                        console=console,
                        session_id=session_id,
                        execution_output=term_out,
                        additional_instructions=additional_instructions,
                        api_keys=metadata_keys,  # Pass client LLM keys
                        auth_headers=auth_headers,  # Pass Weco key if logged in
                        timeout=timeout,
                    )

                    summary_panel.update_token_counts(usage=eval_and_next_solution_response["usage"])
                    status_response = get_optimization_session_status(
                        console=console,
                        session_id=session_id,
                        include_history=True,
                        auth_headers=auth_headers,
                        timeout=timeout,
                    )
                    tree_panel.build_metric_tree(nodes=status_response["history"])

                    if status_response["best_result"] is not None:
                        best_solution_node = Node(
                            id=status_response["best_result"]["solution_id"],
                            parent_id=status_response["best_result"]["parent_id"],
                            code=status_response["best_result"]["code"],
                            metric=status_response["best_result"]["metric_value"],
                            is_buggy=status_response["best_result"]["is_buggy"],
                        )

                    if is_last_evaluation or eval_and_next_solution_response.get("is_done", False):
                        summary_panel.set_step(step=steps)
                        solution_panels.update(current_node=None, best_node=best_solution_node)
                        _, best_solution_panel = solution_panels.get_display(current_step=steps)
                        smooth_update(
                            live=live,
                            layout=layout,
                            sections_to_update=[
                                ("summary", summary_panel.get_display()),
                                ("tree", tree_panel.get_display()),
                                ("best_solution", best_solution_panel),
                                ("current_solution", Panel("Optimization Complete", title="Current Solution")),
                                ("plan", Panel("Optimization Complete", title="Plan")),
                            ],
                            transition_delay=0.1,
                        )
                        break

                    # Prepare for next iteration
                    next_solution_code = eval_and_next_solution_response["code"]
                    write_to_path(fp=runs_dir / f"step_{step + 1}.py", content=next_solution_code)
                    write_to_path(fp=source_fp, content=next_solution_code)

                    summary_panel.set_step(step=step + 1)
                    plan_panel.update(plan=eval_and_next_solution_response["plan"])
                    tree_panel.set_unevaluated_node(node_id=eval_and_next_solution_response["solution_id"])

                    current_solution_node = None
                    for node_data in status_response["history"]:
                        if node_data["solution_id"] == eval_and_next_solution_response["solution_id"]:
                            current_solution_node = Node(
                                id=node_data["solution_id"],
                                parent_id=node_data["parent_id"],
                                code=node_data["code"],
                                metric=node_data["metric_value"],
                                is_buggy=node_data["is_buggy"],
                            )
                            break
                    if current_solution_node is None:
                        print(
                            f"Warning: Could not find newly generated solution {eval_and_next_solution_response['solution_id']} in history."
                        )
                        current_solution_node = Node(
                            id=eval_and_next_solution_response["solution_id"],
                            code=next_solution_code,
                            parent_id=None,
                            metric=None,
                            is_buggy=False,
                        )

                    solution_panels.update(current_node=current_solution_node, best_node=best_solution_node)
                    current_solution_panel, best_solution_panel = solution_panels.get_display(current_step=step + 1)

                    eval_output_panel.clear()
                    smooth_update(
                        live=live,
                        layout=layout,
                        sections_to_update=[
                            ("summary", summary_panel.get_display()),
                            ("plan", plan_panel.get_display()),
                            ("tree", tree_panel.get_display()),
                            ("current_solution", current_solution_panel),
                            ("best_solution", best_solution_panel),
                            ("eval_output", eval_output_panel.get_display()),
                        ],
                        transition_delay=0.08,
                    )
                    # term_out for next iteration is prepared by run_evaluation at start of loop

                # --- Post-Loop Processing ---
                if best_solution_node:
                    solution_panels.update(current_node=None, best_node=best_solution_node)
                    _, best_solution_panel = solution_panels.get_display(current_step=steps)
                else:
                    best_solution_panel = Panel(
                        "[italic]No improvement found.[/]", title=f"Best Solution ({metric_name})", border_style="dim"
                    )

                end_optimization_layout["summary"].update(summary_panel.get_display())
                end_optimization_layout["tree"].update(tree_panel.get_display())
                end_optimization_layout["best_solution"].update(best_solution_panel)

                best_solution_code = best_solution_node.code if best_solution_node else None
                best_solution_score = best_solution_node.metric if best_solution_node else None

                if best_solution_code is None or best_solution_score is None:
                    best_solution_content = (
                        f"# Weco could not find a better solution\n\n{read_from_path(fp=runs_copy_source_fp, is_json=False)}"
                    )
                else:
                    best_score_str = (
                        format_number(best_solution_score)
                        if best_solution_score is not None and isinstance(best_solution_score, (int, float))
                        else "N/A"
                    )
                    best_solution_content = (
                        f"# Best solution from Weco with a score of {best_score_str}\n\n{best_solution_code}"
                    )

                write_to_path(fp=runs_dir / "best.py", content=best_solution_content)
                write_to_path(fp=source_fp, content=best_solution_content)

            console.print(end_optimization_layout)
            # --- End of the main execution block within try ---

        except Exception as e:
            console.print(Panel(f"[bold red]Error: {str(e)}", title="[bold red]Error", border_style="red"))
            # Print traceback for debugging
            console.print_exception(show_locals=True)
            sys.exit(1)

    # Removed the final 'else' block as commands are now required by subparsers
    # else:
    #      parser.print_help()
    #      sys.exit(1)
