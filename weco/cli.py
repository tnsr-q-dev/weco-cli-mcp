import argparse
import sys
import pathlib
import math
import time
import requests
import webbrowser
import threading
import signal
import traceback
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.traceback import install
from rich.prompt import Prompt
from .api import (
    start_optimization_run,
    evaluate_feedback_then_suggest_next_solution,
    get_optimization_run_status,
    handle_api_error,
    send_heartbeat,
    report_termination,
)

from . import __base_url__
from .auth import load_weco_api_key, save_api_key, clear_api_key
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
    check_for_cli_updates,
)

install(show_locals=True)
console = Console()

# --- Global variable for heartbeat thread ---
heartbeat_thread = None
stop_heartbeat_event = threading.Event()
current_run_id_for_heartbeat = None
current_auth_headers_for_heartbeat = {}


# --- Heartbeat Sender Class ---
class HeartbeatSender(threading.Thread):
    def __init__(self, run_id: str, auth_headers: dict, stop_event: threading.Event, interval: int = 30):
        super().__init__(daemon=True)  # Daemon thread exits when main thread exits
        self.run_id = run_id
        self.auth_headers = auth_headers
        self.interval = interval
        self.stop_event = stop_event

    def run(self):
        try:
            while not self.stop_event.is_set():
                if not send_heartbeat(self.run_id, self.auth_headers):
                    # send_heartbeat itself prints errors to stderr if it returns False
                    # No explicit HeartbeatSender log needed here unless more detail is desired for a False return
                    pass  # Continue trying as per original logic

                if self.stop_event.is_set():  # Check before waiting for responsiveness
                    break

                self.stop_event.wait(self.interval)  # Wait for interval or stop signal

        except Exception as e:
            # Catch any unexpected error in the loop to prevent silent thread death
            print(f"[ERROR HeartbeatSender] Unhandled exception in run loop for run {self.run_id}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # The loop will break due to the exception, and thread will terminate via finally.


# --- Signal Handling ---
def signal_handler(signum, frame):
    signal_name = signal.Signals(signum).name
    console.print(f"\n[bold yellow]Termination signal ({signal_name}) received. Shutting down...[/]")

    # Stop heartbeat thread
    stop_heartbeat_event.set()
    if heartbeat_thread and heartbeat_thread.is_alive():
        heartbeat_thread.join(timeout=2)  # Give it a moment to stop

    # Report termination (best effort)
    if current_run_id_for_heartbeat:
        report_termination(
            run_id=current_run_id_for_heartbeat,
            status_update="terminated",
            reason=f"user_terminated_{signal_name.lower()}",
            details=f"Process terminated by signal {signal_name} ({signum}).",
            auth_headers=current_auth_headers_for_heartbeat,
            timeout=3,
        )

    # Exit gracefully
    sys.exit(0)


def perform_login(console: Console):
    """Handles the device login flow."""
    try:
        # 1. Initiate device login
        console.print("Initiating login...")
        init_response = requests.post(f"{__base_url__}/auth/device/initiate")
        init_response.raise_for_status()
        init_data = init_response.json()

        device_code = init_data["device_code"]
        verification_uri = init_data["verification_uri"]
        expires_in = init_data["expires_in"]
        interval = init_data["interval"]

        # 2. Display instructions
        console.print("\n[bold yellow]Action Required:[/]")
        console.print("Please open the following URL in your browser to authenticate:")
        console.print(f"[link={verification_uri}]{verification_uri}[/link]")
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
                        f"{__base_url__}/auth/device/token",
                        json={"grant_type": "urn:ietf:params:oauth:grant-type:device_code", "device_code": device_code},
                    )

                    # Check for 202 Accepted - Authorization Pending
                    if token_response.status_code == 202:
                        token_data = token_response.json()
                        if token_data.get("error") == "authorization_pending":
                            live_status.update("Waiting... (authorization pending)")
                            continue  # Continue polling
                        else:
                            # Unexpected 202 response format
                            console.print(f"\n[bold red]Error:[/] Received unexpected 202 response: {token_data}")
                            return False
                    # Check for standard OAuth2 errors (often 400 Bad Request)
                    elif token_response.status_code == 400:
                        token_data = token_response.json()
                        error_code = token_data.get("error", "unknown_error")
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
                        print(token_data)
                        return False
                except requests.exceptions.RequestException as e:
                    # Handle network errors during polling gracefully
                    live_status.update("Waiting... (network error, retrying)")
                    console.print(f"\n[bold yellow]Warning:[/] Network error during polling: {e}. Retrying...")
                    time.sleep(interval * 2)  # Simple backoff
    except requests.exceptions.HTTPError as e:
        handle_api_error(e, console)
    except requests.exceptions.RequestException as e:
        # Catch other request errors
        console.print(f"\n[bold red]Network Error:[/] {e}")
        return False
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred during login:[/] {e}")
        return False


def main() -> None:
    """Main function for the Weco CLI."""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # --- Perform Update Check ---
    from . import __pkg_version__

    check_for_cli_updates(__pkg_version__)

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description="[bold cyan]Weco CLI[/]", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # --- Run Command ---
    run_parser = subparsers.add_parser(
        "run", help="Run code optimization", formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    # Add arguments specific to the 'run' command to the run_parser
    run_parser.add_argument(
        "-s",
        "--source",
        type=str,
        required=True,
        help="Path to the source code file that will be optimized (e.g., `optimize.py`)",
    )
    run_parser.add_argument(
        "-c",
        "--eval-command",
        type=str,
        required=True,
        help="Command to run for evaluation (e.g. 'python eval.py --arg1=val1').",
    )
    run_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        required=True,
        help="Metric to optimize (e.g. 'accuracy', 'loss', 'f1_score') that is printed to the terminal by the eval command.",
    )
    run_parser.add_argument(
        "-g",
        "--goal",
        type=str,
        choices=["maximize", "max", "minimize", "min"],
        required=True,
        help="Specify 'maximize'/'max' to maximize the metric or 'minimize'/'min' to minimize it.",
    )
    run_parser.add_argument("-n", "--steps", type=int, default=100, help="Number of steps to run. Defaults to 100.")
    run_parser.add_argument(
        "-M",
        "--model",
        type=str,
        default=None,
        help="Model to use for optimization. Defaults to `o4-mini` when `OPENAI_API_KEY` is set, `claude-3-7-sonnet-20250219` when `ANTHROPIC_API_KEY` is set, and `gemini-2.5-pro-exp-03-25` when `GEMINI_API_KEY` is set. When multiple keys are set, the priority is `OPENAI_API_KEY` > `ANTHROPIC_API_KEY` > `GEMINI_API_KEY`.",
    )
    run_parser.add_argument(
        "-l", "--log-dir", type=str, default=".runs", help="Directory to store logs and results. Defaults to `.runs`."
    )
    run_parser.add_argument(
        "-i",
        "--additional-instructions",
        default=None,
        type=str,
        help="Description of additional instruction or path to a file containing additional instructions. Defaults to None.",
    )

    _ = subparsers.add_parser("logout", help="Log out from Weco and clear saved API key.")
    args = parser.parse_args()

    if args.command == "logout":
        clear_api_key()
        sys.exit(0)
    elif args.command == "run":
        global heartbeat_thread, current_run_id_for_heartbeat, current_auth_headers_for_heartbeat  # Allow modification of globals
        run_id = None  # Initialize run_id (we receive this from the API after starting the run)
        optimization_completed_normally = False
        user_stop_requested_flag = False
        weco_api_key = load_weco_api_key()
        llm_api_keys = read_api_keys_from_env()  # Read keys from client environment

        # --- Login/Authentication Handling ---
        if not weco_api_key:
            login_choice = Prompt.ask(
                "Log in to Weco to save run history or use anonymously? ([bold]L[/]ogin / [bold]S[/]kip)",
                choices=["l", "s"],
                default="s",
            ).lower()
            if login_choice == "l":
                console.print("[cyan]Starting login process...[/]")
                if not perform_login(console):
                    console.print("[bold red]Login process failed or was cancelled.[/]")
                    sys.exit(1)
                weco_api_key = load_weco_api_key()
                if not weco_api_key:
                    console.print("[bold red]Error: Login completed but failed to retrieve API key.[/]")
                    sys.exit(1)
            elif login_choice == "s":
                console.print("[yellow]Proceeding anonymously. LLM API keys must be provided via environment variables.[/]")
                if not llm_api_keys:
                    console.print(
                        "[bold red]Error:[/] No LLM API keys found in environment (e.g., OPENAI_API_KEY). Cannot proceed anonymously."
                    )
                    sys.exit(1)

        auth_headers = {}
        if weco_api_key:
            auth_headers["Authorization"] = f"Bearer {weco_api_key}"
        current_auth_headers_for_heartbeat = auth_headers  # Store for signal handler

        # --- Main Run Logic ---
        try:
            # --- Read Command Line Arguments ---
            evaluation_command = args.eval_command
            metric_name = args.metric
            maximize = args.goal in ["maximize", "max"]
            steps = args.steps
            # Determine the model to use
            if args.model is None:
                if "OPENAI_API_KEY" in llm_api_keys:
                    args.model = "o4-mini"
                elif "ANTHROPIC_API_KEY" in llm_api_keys:
                    args.model = "claude-3-7-sonnet-20250219"
                elif "GEMINI_API_KEY" in llm_api_keys:
                    args.model = "gemini-2.5-pro-exp-03-25"
                else:
                    raise ValueError(
                        "No LLM API keys found in environment. Please set one of the following: OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY."
                    )
            code_generator_config = {"model": args.model}
            evaluator_config = {"model": args.model, "include_analysis": True}
            search_policy_config = {
                "num_drafts": max(1, math.ceil(0.15 * steps)),
                "debug_prob": 0.5,
                "max_debug_depth": max(1, math.ceil(0.1 * steps)),
            }
            timeout = 800
            additional_instructions = read_additional_instructions(additional_instructions=args.additional_instructions)
            source_fp = pathlib.Path(args.source)
            source_code = read_from_path(fp=source_fp, is_json=False)

            # --- Panel Initialization ---
            summary_panel = SummaryPanel(
                maximize=maximize, metric_name=metric_name, total_steps=steps, model=args.model, runs_dir=args.log_dir
            )
            plan_panel = PlanPanel()
            solution_panels = SolutionPanels(metric_name=metric_name, source_fp=source_fp)
            eval_output_panel = EvaluationOutputPanel()
            tree_panel = MetricTreePanel(maximize=maximize)
            layout = create_optimization_layout()
            end_optimization_layout = create_end_optimization_layout()

            # --- Start Optimization Run ---
            run_response = start_optimization_run(
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
                api_keys=llm_api_keys,
                auth_headers=auth_headers,
                timeout=timeout,
            )
            run_id = run_response["run_id"]
            current_run_id_for_heartbeat = run_id

            # --- Start Heartbeat Thread ---
            stop_heartbeat_event.clear()
            heartbeat_thread = HeartbeatSender(run_id, auth_headers, stop_heartbeat_event)
            heartbeat_thread.start()

            # --- Live Update Loop ---
            refresh_rate = 4
            with Live(layout, refresh_per_second=refresh_rate, screen=True) as live:
                # Define the runs directory (.runs/<run-id>) to store logs and results
                runs_dir = pathlib.Path(args.log_dir) / run_id
                runs_dir.mkdir(parents=True, exist_ok=True)
                # Write the initial code string to the logs
                write_to_path(fp=runs_dir / f"step_0{source_fp.suffix}", content=run_response["code"])
                # Write the initial code string to the source file path
                write_to_path(fp=source_fp, content=run_response["code"])

                # Update the panels with the initial solution
                summary_panel.set_run_id(run_id=run_id)  # Add run id now that we have it
                # Set the step of the progress bar
                summary_panel.set_step(step=0)
                # Update the token counts
                summary_panel.update_token_counts(usage=run_response["usage"])
                plan_panel.update(plan=run_response["plan"])
                # Build the metric tree
                tree_panel.build_metric_tree(
                    nodes=[
                        {
                            "solution_id": run_response["solution_id"],
                            "parent_id": None,
                            "code": run_response["code"],
                            "step": 0,
                            "metric_value": None,
                            "is_buggy": False,
                        }
                    ]
                )
                # Set the current solution as unevaluated since we haven't run the evaluation function and fed it back to the model yet
                tree_panel.set_unevaluated_node(node_id=run_response["solution_id"])
                # Update the solution panels with the initial solution and get the panel displays
                solution_panels.update(
                    current_node=Node(
                        id=run_response["solution_id"], parent_id=None, code=run_response["code"], metric=None, is_buggy=False
                    ),
                    best_node=None,
                )
                current_solution_panel, best_solution_panel = solution_panels.get_display(current_step=0)
                # Update the live layout with the initial solution panels
                smooth_update(
                    live=live,
                    layout=layout,
                    sections_to_update=[
                        ("summary", summary_panel.get_display()),
                        ("plan", plan_panel.get_display()),
                        ("tree", tree_panel.get_display(is_done=False)),
                        ("current_solution", current_solution_panel),
                        ("best_solution", best_solution_panel),
                        ("eval_output", eval_output_panel.get_display()),
                    ],
                    transition_delay=0.1,
                )

                # Run evaluation on the initial solution
                term_out = run_evaluation(eval_command=args.eval_command)
                # Update the evaluation output panel
                eval_output_panel.update(output=term_out)
                smooth_update(
                    live=live,
                    layout=layout,
                    sections_to_update=[("eval_output", eval_output_panel.get_display())],
                    transition_delay=0.1,
                )

                # Starting from step 1 to steps (inclusive) because the baseline solution is step 0, so we want to optimize for steps worth of steps
                for step in range(1, steps + 1):
                    # Re-read instructions from the original source (file path or string) BEFORE each suggest call
                    current_additional_instructions = read_additional_instructions(
                        additional_instructions=args.additional_instructions
                    )
                    if run_id:
                        try:
                            current_status_response = get_optimization_run_status(
                                run_id=run_id, include_history=False, timeout=30, auth_headers=auth_headers
                            )
                            current_run_status_val = current_status_response.get("status")
                            if current_run_status_val == "stopping":
                                console.print("\n[bold yellow]Stop request received. Terminating run gracefully...[/]")
                                user_stop_requested_flag = True
                                break
                        except requests.exceptions.RequestException as e:
                            console.print(
                                f"\n[bold red]Warning: Could not check run status: {e}. Continuing optimization...[/]"
                            )
                        except Exception as e:
                            console.print(
                                f"\n[bold red]Warning: Error checking run status: {e}. Continuing optimization...[/]"
                            )

                    # Send feedback and get next suggestion
                    eval_and_next_solution_response = evaluate_feedback_then_suggest_next_solution(
                        run_id=run_id,
                        execution_output=term_out,
                        additional_instructions=current_additional_instructions,
                        api_keys=llm_api_keys,
                        auth_headers=auth_headers,
                        timeout=timeout,
                    )
                    # Save next solution (.runs/<run-id>/step_<step>.<extension>)
                    write_to_path(
                        fp=runs_dir / f"step_{step}{source_fp.suffix}", content=eval_and_next_solution_response["code"]
                    )
                    # Write the next solution to the source file
                    write_to_path(fp=source_fp, content=eval_and_next_solution_response["code"])
                    status_response = get_optimization_run_status(
                        run_id=run_id, include_history=True, timeout=timeout, auth_headers=auth_headers
                    )
                    # Update the step of the progress bar, token counts, plan and metric tree
                    summary_panel.set_step(step=step)
                    summary_panel.update_token_counts(usage=eval_and_next_solution_response["usage"])
                    plan_panel.update(plan=eval_and_next_solution_response["plan"])

                    nodes_list_from_status = status_response.get("nodes")
                    tree_panel.build_metric_tree(nodes=nodes_list_from_status if nodes_list_from_status is not None else [])
                    tree_panel.set_unevaluated_node(node_id=eval_and_next_solution_response["solution_id"])

                    # Update the solution panels with the next solution and best solution (and score)
                    # Figure out if we have a best solution so far
                    if status_response["best_result"] is not None:
                        best_solution_node = Node(
                            id=status_response["best_result"]["solution_id"],
                            parent_id=status_response["best_result"]["parent_id"],
                            code=status_response["best_result"]["code"],
                            metric=status_response["best_result"]["metric_value"],
                            is_buggy=status_response["best_result"]["is_buggy"],
                        )
                    else:
                        best_solution_node = None

                    current_solution_node = None
                    if status_response.get("nodes"):
                        for node_data in status_response["nodes"]:
                            if node_data["solution_id"] == eval_and_next_solution_response["solution_id"]:
                                current_solution_node = Node(
                                    id=node_data["solution_id"],
                                    parent_id=node_data["parent_id"],
                                    code=node_data["code"],
                                    metric=node_data["metric_value"],
                                    is_buggy=node_data["is_buggy"],
                                )
                    if current_solution_node is None:
                        raise ValueError("Current solution node not found in nodes list from status response")

                    # Update the solution panels with the current and best solution
                    solution_panels.update(current_node=current_solution_node, best_node=best_solution_node)
                    current_solution_panel, best_solution_panel = solution_panels.get_display(current_step=step)
                    # Clear evaluation output since we are running a evaluation on a new solution
                    eval_output_panel.clear()
                    smooth_update(
                        live=live,
                        layout=layout,
                        sections_to_update=[
                            ("summary", summary_panel.get_display()),
                            ("plan", plan_panel.get_display()),
                            ("tree", tree_panel.get_display(is_done=False)),
                            ("current_solution", current_solution_panel),
                            ("best_solution", best_solution_panel),
                            ("eval_output", eval_output_panel.get_display()),
                        ],
                        transition_delay=0.08,  # Slightly longer delay for more noticeable transitions
                    )
                    term_out = run_evaluation(eval_command=args.eval_command)
                    eval_output_panel.update(output=term_out)
                    smooth_update(
                        live=live,
                        layout=layout,
                        sections_to_update=[("eval_output", eval_output_panel.get_display())],
                        transition_delay=0.1,
                    )

                if not user_stop_requested_flag:
                    # Re-read instructions from the original source (file path or string) BEFORE each suggest call
                    current_additional_instructions = read_additional_instructions(
                        additional_instructions=args.additional_instructions
                    )
                    # Evaluate the final solution thats been generated
                    eval_and_next_solution_response = evaluate_feedback_then_suggest_next_solution(
                        run_id=run_id,
                        execution_output=term_out,
                        additional_instructions=current_additional_instructions,
                        api_keys=llm_api_keys,
                        timeout=timeout,
                        auth_headers=auth_headers,
                    )
                    summary_panel.set_step(step=steps)
                    summary_panel.update_token_counts(usage=eval_and_next_solution_response["usage"])
                    status_response = get_optimization_run_status(
                        run_id=run_id, include_history=True, timeout=timeout, auth_headers=auth_headers
                    )
                    # No need to update the plan panel since we have finished the optimization
                    # Get the optimization run status for
                    # the best solution, its score, and the history to plot the tree
                    nodes_list_from_status_final = status_response.get("nodes")
                    tree_panel.build_metric_tree(
                        nodes=nodes_list_from_status_final if nodes_list_from_status_final is not None else []
                    )
                    # No need to set any solution to unevaluated since we have finished the optimization
                    # and all solutions have been evaluated
                    # No neeed to update the current solution panel since we have finished the optimization
                    # We only need to update the best solution panel
                    # Figure out if we have a best solution so far
                    if status_response["best_result"] is not None:
                        best_solution_node = Node(
                            id=status_response["best_result"]["solution_id"],
                            parent_id=status_response["best_result"]["parent_id"],
                            code=status_response["best_result"]["code"],
                            metric=status_response["best_result"]["metric_value"],
                            is_buggy=status_response["best_result"]["is_buggy"],
                        )
                    else:
                        best_solution_node = None
                    solution_panels.update(current_node=None, best_node=best_solution_node)
                    _, best_solution_panel = solution_panels.get_display(current_step=steps)
                    # Update the end optimization layout
                    final_message = (
                        f"{summary_panel.metric_name.capitalize()} {'maximized' if summary_panel.maximize else 'minimized'}! Best solution {summary_panel.metric_name.lower()} = [green]{status_response['best_result']['metric_value']}[/] 🏆"
                        if best_solution_node is not None and best_solution_node.metric is not None
                        else "[red] No valid solution found.[/]"
                    )
                    end_optimization_layout["summary"].update(summary_panel.get_display(final_message=final_message))
                    end_optimization_layout["tree"].update(tree_panel.get_display(is_done=True))
                    end_optimization_layout["best_solution"].update(best_solution_panel)

                    # Save optimization results
                    # If the best solution does not exist or is has not been measured at the end of the optimization
                    # save the original solution as the best solution
                    if best_solution_node is not None:
                        best_solution_code = best_solution_node.code
                        best_solution_score = best_solution_node.metric
                    else:
                        best_solution_code = None
                        best_solution_score = None

                    if best_solution_code is None or best_solution_score is None:
                        best_solution_content = f"# Weco could not find a better solution\n\n{read_from_path(fp=runs_dir / f'step_0{source_fp.suffix}', is_json=False)}"
                    else:
                        # Format score for the comment
                        best_score_str = (
                            format_number(best_solution_score)
                            if best_solution_score is not None and isinstance(best_solution_score, (int, float))
                            else "N/A"
                        )
                        best_solution_content = (
                            f"# Best solution from Weco with a score of {best_score_str}\n\n{best_solution_code}"
                        )
                    # Save best solution to .runs/<run-id>/best.<extension>
                    write_to_path(fp=runs_dir / f"best{source_fp.suffix}", content=best_solution_content)
                    # write the best solution to the source file
                    write_to_path(fp=source_fp, content=best_solution_content)
                    # Mark as completed normally for the finally block
                    optimization_completed_normally = True
                    console.print(end_optimization_layout)

        except Exception as e:
            # Catch errors during the main optimization loop or setup
            try:
                error_message = e.response.json()["detail"]
            except Exception:
                error_message = str(e)
            console.print(Panel(f"[bold red]Error: {error_message}", title="[bold red]Optimization Error", border_style="red"))
            # Ensure optimization_completed_normally is False
            optimization_completed_normally = False
            error_details = traceback.format_exc()
            exit_code = 1
        finally:
            # This block runs whether the try block completed normally or raised an exception
            # Stop heartbeat thread
            stop_heartbeat_event.set()
            if heartbeat_thread and heartbeat_thread.is_alive():
                heartbeat_thread.join(timeout=2)

            # Report final status if a run was started
            if run_id:
                final_status_update = "unknown"
                final_reason_code = "unknown_termination"
                final_details = None
                if optimization_completed_normally:
                    final_status_update = "completed"
                    final_reason_code = "completed_successfully"
                elif user_stop_requested_flag:
                    final_status_update = "terminated"
                    final_reason_code = "user_requested_stop"
                    final_details = "Run stopped by user request via dashboard."
                else:
                    final_status_update = "error"
                    final_reason_code = "error_cli_internal"
                    if "error_details" in locals():
                        final_details = locals()["error_details"]
                    elif "e" in locals() and isinstance(locals()["e"], Exception):
                        final_details = traceback.format_exc()
                    else:
                        final_details = "CLI terminated unexpectedly without a specific exception captured."
                # Keep default 'unknown' if we somehow end up here without error/completion/signal
                # Avoid reporting if terminated by signal handler (already reported)
                # Check a flag or rely on status not being 'unknown'
                if final_status_update != "unknown":
                    report_termination(
                        run_id=run_id,
                        status_update=final_status_update,
                        reason=final_reason_code,
                        details=final_details,
                        auth_headers=current_auth_headers_for_heartbeat,
                    )
            if optimization_completed_normally:
                sys.exit(0)
            elif user_stop_requested_flag:
                console.print("[yellow]Run terminated by user request.[/]")
                sys.exit(0)
            else:
                sys.exit(locals().get("exit_code", 1))
