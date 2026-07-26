# Tutorial: evaluating a merged model

Merging is fast and cheap, which makes it tempting to trust. Don't — a merge is an
**unvalidated candidate**. This tutorial is about measuring whether it actually
helped, and about greedy soups, which *require* an evaluator.

## Why evaluation is mandatory

Merging offers no guarantee of quality, calibration, safety, or bias behavior
(see [Limitations](../limitations.md)). The only way to know is to measure on data
the merge never touched.

## Pick task-appropriate metrics

- **Language models**: perplexity on held-out text; task accuracy / F1 for
  classification; generation-quality metrics for open-ended tasks.
- **Always**: compare against strong **baselines** — the best single input model
  and, where relevant, an output **ensemble** of the inputs.
- **Report uncertainty**: a single number without variance across seeds/splits is
  not evidence.

## Avoid data leakage

- Use a **held-out** test set the models were not trained on.
- For **greedy soups**, keep the **selection set** (used to score candidates)
  strictly separate from the **final test set** (used to report). Selecting a
  subset by a metric fits to that metric; reusing the same data inflates results.

## Safety and bias

Averaging weights can weaken alignment. Re-run your safety and bias evaluations on
the merged model; do not assume they carry over from the inputs.

## Wiring an evaluator into greedy soup

A greedy soup needs a function that scores a written checkpoint. **Callable**
(Python):

```python
# my_eval.py
from pathlib import Path

def score(model_path: Path) -> float:
    # load the model from model_path, run it on your SELECTION set,
    # and return a metric where higher is better (use direction: minimize for loss).
    ...
    return accuracy
```

```yaml
algorithm: {type: greedy_soup}
greedy:
  direction: maximize
  evaluator: {type: callable, callable: "my_eval:score"}
```

**Command** (any language), printing `{"score": <float>}`:

```yaml
greedy:
  direction: maximize
  evaluator:
    type: command
    command: ["python", "my_eval_cli.py", "{model_path}"]
    metric_key: score
    timeout: 600
```

`{model_path}` is passed as a single argv element (no shell), so paths with spaces
are safe.

## Interpreting the greedy report

```bash
model-merger merge greedy.yaml --json | jq '.algorithm_params, .greedy_history'
```

`accepted` lists the chosen models; `greedy_history` shows each accept/reject
decision and its score. If greedy accepted only the seed, no candidate improved
the metric — that is a real (if disappointing) result, not a bug.

## The workflow

1. Merge candidates.
2. Evaluate each on a held-out set against baselines.
3. If using greedy selection, keep selection and test data disjoint.
4. Run safety/bias checks.
5. Only then decide whether the merged model is worth keeping.
