import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from pathlib import Path


class InvalidSubmissionError(Exception):
    """
    A custom exception for when the agent submission cannot be graded.
    """

    pass


def prepare_for_accuracy_metric(submission: pd.DataFrame, answers: pd.DataFrame, target_column: str, id_column: str) -> dict:
    # Answers checks
    assert target_column in answers.columns, f"Answers must have a `{target_column}` column"
    assert id_column in answers.columns, f"Answers must have a `{id_column}` column"

    # Submission checks
    if len(submission) != len(answers):
        raise InvalidSubmissionError("Submission must have the same length as the answers.")
    if target_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have a `{target_column}` column")
    if id_column not in submission.columns:
        raise InvalidSubmissionError(f"Submission must have a `{id_column}` column")

    # sort on id to ensure correct order
    submission = submission.sort_values(id_column)
    answers = answers.sort_values(id_column)

    if (submission[id_column].values != answers[id_column].values).any():
        raise InvalidSubmissionError(f"Submission and Answers `{id_column}`'s do not match")

    y_pred = submission[target_column].to_numpy()
    y_true = answers[target_column].to_numpy()

    return {"y_true": y_true, "y_pred": y_pred}


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    accuracy_inputs = prepare_for_accuracy_metric(
        submission=submission, answers=answers, target_column="Transported", id_column="PassengerId"
    )
    return accuracy_score(**accuracy_inputs)


if __name__ == "__main__":
    # Get the directory where the script is located
    script_dir = Path(__file__).resolve().parent
    # The ground truth answers are now in private/test.csv
    answers_path = script_dir / "private" / "test.csv"
    # Assume the agent's submission is saved here
    submission_path = script_dir / "submission.csv"

    # Check if files exist before proceeding
    if not answers_path.exists():
        print(f"Error: Answers file not found at {answers_path}")  # Updated path in error message
        sys.exit(1)

    if not submission_path.exists():
        print(f"Error: Submission file not found at {submission_path}")
        sys.exit(1)

    submission = pd.read_csv(submission_path)
    # Read answers from the updated path
    answers = pd.read_csv(answers_path)

    # Calculate and print the grade
    score = grade(submission, answers)
    print(f"accuracy: {score}")
