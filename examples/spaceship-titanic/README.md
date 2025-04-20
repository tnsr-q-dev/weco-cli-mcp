# Example: Optimizing a Kaggle Classification Model (Spaceship Titanic)

This example demonstrates using Weco to optimize a Python script designed for the [Spaceship Titanic Kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic/overview). The goal is to improve the model's `accuracy` metric by modifying the feature engineering and modeling steps within the `optimize.py` script.

This example uses the `README.md` file (this file) to provide additional instructions to the LLM.

## Setup

1.  Ensure you are in the `examples/spaceship-titanic` directory.
2.  **Kaggle Credentials:** You need your Kaggle API credentials (`kaggle.json`) configured to download the competition dataset. Place the `kaggle.json` file in `~/.kaggle/` or set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables. See [Kaggle API documentation](https://github.com/Kaggle/kaggle-api#api-credentials) for details.
3.  **Install Dependencies:** Install the required Python packages:
    ```bash
    pip install -r requirements-test.txt
    ```
4.  **Prepare Data:** Run the utility script once to download the dataset from Kaggle and place it in the expected `public/` and `private/` subdirectories:
    ```bash
    python utils.py
    ```
    After running `utils.py`, your directory structure should look like this:
    ```
    .
    ├── baseline.py
    ├── evaluate.py
    ├── optimize.py
    ├── private
    │   └── test.csv
    ├── public
    │   ├── sample_submission.csv
    │   ├── test.csv
    │   └── train.csv
    ├── README.md  # This file
    ├── requirements-test.txt
    └── utils.py
    ```

## Optimization Command

Run the following command to start optimizing the model:

```bash
weco --source optimize.py \
     --eval-command "python optimize.py && python evaluate.py" \
     --metric accuracy \
     --maximize true \
     --steps 10 \
     --model gemini-2.5-pro-exp-03-25 \
     --additional-instructions README.md
```

### Explanation

*   `--source optimize.py`: The script containing the model training and prediction logic to be optimized. It starts identical to `baseline.py`.
*   `--eval-command "python optimize.py && python evaluate.py"`: This is a multi-step evaluation.
    *   `python optimize.py`: Runs the modified script to generate predictions (`submission.csv`).
    *   `python evaluate.py`: Compares the generated `submission.csv` against the ground truth (using the training data as a proxy evaluation set in this example) and prints the `accuracy` metric.
*   `--metric accuracy`: The target metric Weco should optimize.
*   `--maximize true`: Weco aims to increase the accuracy.
*   `--steps 10`: The number of optimization iterations.
*   `--model gemini-2.5-pro-exp-03-25`: The LLM driving the optimization.
*   `--additional-instructions README.md`: Provides this file as context to the LLM, which might include hints about feature engineering techniques, model types to try, or specific data columns to focus on (you can add such instructions to this file if desired).

Weco will iteratively modify the feature engineering or modeling code within `optimize.py`, run the evaluation pipeline, and use the resulting `accuracy` to guide further improvements. The `baseline.py` file is provided as a reference starting point.