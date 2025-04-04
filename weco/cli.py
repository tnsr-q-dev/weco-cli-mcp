import argparse
import sys
import pathlib
import math
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.traceback import install
from .api import start_optimization_session, evaluate_feedback_then_suggest_next_solution, get_optimization_session_status
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


def main() -> None:
    """Main function for the Weco CLI."""
    parser = argparse.ArgumentParser(
        description="[bold cyan]Weco CLI[/]", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=str, required=True, help="Path to the Python source code (e.g. optimize.py)")
    parser.add_argument(
        "--eval-command", type=str, required=True, help="Command to run for evaluation (e.g. 'python eval.py --arg1=val1')"
    )
    parser.add_argument("--metric", type=str, required=True, help="Metric to optimize")
    parser.add_argument(
        "--maximize",
        type=str,
        choices=["true", "false"],
        required=True,
        help="Specify 'true' to maximize the metric or 'false' to minimize.",
    )
    parser.add_argument("--steps", type=int, required=True, help="Number of steps to run")
    parser.add_argument("--model", type=str, required=True, help="Model to use for optimization")
    parser.add_argument(
        "--additional-instructions",
        default=None,
        type=str,
        help="Description of additional instruction or path to a file containing additional instructions",
    )
    args = parser.parse_args()

    try:
        with console.status("[bold green]Loading Modules..."):
            # Define optimization session config
            evaluation_command = args.eval_command
            metric_name = args.metric
            maximize = args.maximize == "true"
            steps = args.steps
            code_generator_config = {"model": args.model}
            evaluator_config = {"model": args.model}
            search_policy_config = {
                "num_drafts": max(1, math.ceil(0.15 * steps)),  # 15% of steps
                "debug_prob": 0.5,
                "max_debug_depth": max(1, math.ceil(0.1 * steps)),  # 10% of steps
            }
            # Read additional instructions
            additional_instructions = read_additional_instructions(additional_instructions=args.additional_instructions)
            # Read source code
            source_fp = pathlib.Path(args.source)
            source_code = read_from_path(fp=source_fp, is_json=False)
            # Read API keys
            api_keys = read_api_keys_from_env()

        # Initialize panels
        summary_panel = SummaryPanel(maximize=maximize, metric_name=metric_name, total_steps=steps, model=args.model)
        plan_panel = PlanPanel()
        solution_panels = SolutionPanels(metric_name=metric_name)
        eval_output_panel = EvaluationOutputPanel()
        tree_panel = MetricTreePanel(maximize=maximize)
        layout = create_optimization_layout()
        end_optimization_layout = create_end_optimization_layout()

        # Start optimization session
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
            api_keys=api_keys,
        )

        # Define the refresh rate
        refresh_rate = 4
        with Live(layout, refresh_per_second=refresh_rate, screen=True) as live:
            # Define the runs directory (.runs/<session-id>)
            session_id = session_response["session_id"]
            runs_dir = pathlib.Path(".runs") / session_id
            runs_dir.mkdir(parents=True, exist_ok=True)

            # Save the original code (.runs/<session-id>/original.py)
            runs_copy_source_fp = runs_dir / "original.py"
            write_to_path(fp=runs_copy_source_fp, content=source_code)

            # Write the code string to the source file path
            # Do this after the original code is saved
            write_to_path(fp=source_fp, content=session_response["code"])

            # Update the panels with the initial solution
            # Add session id now that we have it
            summary_panel.session_id = session_id
            # Set the step of the progress bar
            summary_panel.set_step(step=0)
            # Update the token counts
            summary_panel.update_token_counts(usage=session_response["usage"])
            # Update the plan
            plan_panel.update(plan=session_response["plan"])
            # Build the metric tree
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
            # Set the current solution as unevaluated since we haven't run the evaluation function and fed it back to the model yet
            tree_panel.set_unevaluated_node(node_id=session_response["solution_id"])
            # Update the solution panels with the initial solution and get the panel displays
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

            # Update the entire layout
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
                transition_delay=0.1,  # Slightly longer delay for initial display
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

            for step in range(1, steps):
                # Evaluate the current output and get the next solution
                eval_and_next_solution_response = evaluate_feedback_then_suggest_next_solution(
                    console=console,
                    session_id=session_id,
                    execution_output=term_out,
                    additional_instructions=additional_instructions,
                    api_keys=api_keys,
                )
                # Save next solution (.runs/<session-id>/step_<step>.py)
                write_to_path(fp=runs_dir / f"step_{step}.py", content=eval_and_next_solution_response["code"])

                # Write the next solution to the source file
                write_to_path(fp=source_fp, content=eval_and_next_solution_response["code"])

                # Get the optimization session status for
                # the best solution, its score, and the history to plot the tree
                status_response = get_optimization_session_status(console=console, session_id=session_id, include_history=True)

                # Update the step of the progress bar
                summary_panel.set_step(step=step)
                # Update the token counts
                summary_panel.update_token_counts(usage=eval_and_next_solution_response["usage"])
                # Update the plan
                plan_panel.update(plan=eval_and_next_solution_response["plan"])
                # Build the metric tree
                tree_panel.build_metric_tree(nodes=status_response["history"])
                # Set the current solution as unevaluated since we haven't run the evaluation function and fed it back to the model yet
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

                # Create a node for the current solution
                current_solution_node = None
                for node in status_response["history"]:
                    if node["solution_id"] == eval_and_next_solution_response["solution_id"]:
                        current_solution_node = Node(
                            id=node["solution_id"],
                            parent_id=node["parent_id"],
                            code=node["code"],
                            metric=node["metric_value"],
                            is_buggy=node["is_buggy"],
                        )
                if current_solution_node is None:
                    raise ValueError("Current solution node not found in history")
                # Update the solution panels with the current and best solution
                solution_panels.update(current_node=current_solution_node, best_node=best_solution_node)
                current_solution_panel, best_solution_panel = solution_panels.get_display(current_step=step)

                # Clear evaluation output since we are running a evaluation on a new solution
                eval_output_panel.clear()

                # Update displays with smooth transitions
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
                    transition_delay=0.08,  # Slightly longer delay for more noticeable transitions
                )

                # Run evaluation on the current solution
                term_out = run_evaluation(eval_command=args.eval_command)
                eval_output_panel.update(output=term_out)

                # Update evaluation output with a smooth transition
                smooth_update(
                    live=live,
                    layout=layout,
                    sections_to_update=[("eval_output", eval_output_panel.get_display())],
                    transition_delay=0.1,  # Slightly longer delay for evaluation results
                )

            # Ensure we pass evaluation results for the last step's generated solution
            eval_and_next_solution_response = evaluate_feedback_then_suggest_next_solution(
                console=console,
                session_id=session_id,
                execution_output=term_out,
                additional_instructions=additional_instructions,
                api_keys=api_keys,
            )

            # Update the progress bar
            summary_panel.set_step(step=steps)
            # Update the token counts
            summary_panel.update_token_counts(usage=eval_and_next_solution_response["usage"])
            # No need to update the plan panel since we have finished the optimization
            # Get the optimization session status for
            # the best solution, its score, and the history to plot the tree
            status_response = get_optimization_session_status(console=console, session_id=session_id, include_history=True)
            # Build the metric tree
            tree_panel.build_metric_tree(nodes=status_response["history"])
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
            end_optimization_layout["summary"].update(summary_panel.get_display())
            end_optimization_layout["tree"].update(tree_panel.get_display())
            end_optimization_layout["best_solution"].update(best_solution_panel)

            # Save optimization results
            # If the best solution does not exist or is has not been measured at the end of the optimization
            # save the original solution as the best solution
            best_solution_code = best_solution_node.code
            best_solution_score = best_solution_node.metric
            if best_solution_code is None or best_solution_score is None:
                best_solution_content = (
                    f"# Weco could not find a better solution\n\n{read_from_path(fp=runs_copy_source_fp, is_json=False)}"
                )
            else:
                # Format score for the comment
                best_score_str = (
                    format_number(best_solution_score)
                    if best_solution_score is not None and isinstance(best_solution_score, (int, float))
                    else "N/A"
                )
                best_solution_content = f"# Best solution from Weco with a score of {best_score_str}\n\n{best_solution_code}"

            # Save best solution to .runs/<session-id>/best.py
            write_to_path(fp=runs_dir / "best.py", content=best_solution_content)

            # write the best solution to the source file
            write_to_path(fp=source_fp, content=best_solution_content)

        console.print(end_optimization_layout)

    except Exception as e:
        console.print(Panel(f"[bold red]Error: {str(e)}", title="[bold red]Error", border_style="red"))
        sys.exit(1)
