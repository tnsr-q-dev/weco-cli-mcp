# Example: Solving a Kaggle Competition (Spaceship Titanic)

This example demonstrates using Weco to optimize a Python script designed for the [Spaceship Titanic Kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic/overview). The goal is to improve the model's `accuracy` metric by directly optimizing the evaluate.py

## Setup

1.  Ensure you are in the `examples/spaceship-titanic` directory.
2.  `pip install weco`
3.  Set up LLM API Key, `export OPENAI_API_KEY="your_key_here"`
4.  **Install Dependencies:** Install the required Python packages:
    ```bash
    pip install -r requirements-test.txt
    ```

## Optimization Command

Run the following command to start optimizing the model:

```bash
weco run --source evaluate.py \
         --eval-command "python evaluate.py --data-dir ./data" \
         --metric accuracy \
         --goal maximize \
         --steps 20 \
         --model o4-mini \
         --additional-instructions "Improve feature engineering, model choice and hyper-parameters."
         --log-dir .runs/spaceship-titanic
```

### Explanation

*   `--source evaluate.py`: The script provides a baseline as root node and directly optimize the evaluate.py
*   `--eval-command "python evaluate.py --data-dir ./data/"`: The weco agent will run the `evaluate.py` and update it.
    *   [optional] `--data-dir`: path to the train and test data.
    *   [optional] `--seed`: Seed for reproduce the experiment.
*   `--metric accuracy`: The target metric Weco should optimize.
*   `--goal maximize`: Weco aims to increase the accuracy.
*   `--steps 10`: The number of optimization iterations.
*   `--model gemini-2.5-pro-exp-03-25`: The LLM driving the optimization.
*   `--additional-instructions "Improve feature engineering, model choice and hyper-parameters."`: A simple instruction for model improvement or you can put the path to [`comptition_description.md`](./competition_description.md) within the repo to feed the agent more detailed information.

Weco will iteratively modify the feature engineering or modeling code within `evaluate.py`, run the evaluation pipeline, and use the resulting `accuracy` to guide further improvements.
