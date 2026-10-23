"""
Tiny web page that lets the Smart Elevator project run as a Render web service.

It shows the results of the full 20,000-episode training run (saved in results/) and
has a button that trains a small agent live (up to 1,000 episodes) and compares it with
the Random and Rule-based baselines.

Run locally:   python app.py          then open http://127.0.0.1:5000
On Render:     gunicorn app:app --timeout 120 --workers 1
"""

import os

from flask import Flask, request, send_from_directory

from elevator_env import ElevatorEnv, RandomPolicy, rule_based_policy, evaluate
from q_learning_elevator import (QLearningAgent, train, TRAIN_ENV_SEED,
                                 RANDOM_POLICY_SEED, EVAL_SEED, EVAL_EPISODES)

app = Flask(__name__)
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DEMO_EPISODES = 500
MAX_DEMO_EPISODES = 1000      # keeps a live demo short enough for a small server

# Results of the full 20,000-episode run (same numbers as in the README).
FULL_RESULTS = {
    "Random":     {"avg_reward": -916.0, "avg_wait": 25.07, "max_wait": 188, "total_moves": 10757, "delivered": 3500},
    "Rule-based": {"avg_reward": 103.9,  "avg_wait": 4.11,  "max_wait": 39,  "total_moves": 9190,  "delivered": 4256},
    "Q-learning": {"avg_reward": 159.5,  "avg_wait": 4.50,  "max_wait": 54,  "total_moves": 14167, "delivered": 4244},
}


def table_html(results):
    """Turn a {policy name: evaluate() result} dict into an HTML table."""
    rows = ""
    for name, r in results.items():
        rows += (f"<tr><td>{name}</td><td>{r['avg_reward']:.1f}</td><td>{r['avg_wait']:.2f}</td>"
                 f"<td>{r['max_wait']}</td><td>{r['total_moves']}</td><td>{r['delivered']}</td></tr>")
    return ("<table><tr><th>Policy</th><th>Avg reward</th><th>Avg wait</th><th>Max wait</th>"
            f"<th>Total moves</th><th>Delivered</th></tr>{rows}</table>")


def page(title, body):
    """Wrap some HTML in a simple page with a little styling."""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
 body {{ font-family: system-ui, sans-serif; max-width: 820px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }}
 table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
 th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
 th:first-child, td:first-child {{ text-align: left; }}
 img {{ max-width: 100%; }}
 input[type=number] {{ width: 6rem; }}
</style></head><body>{body}</body></html>"""


@app.route("/")
def index():
    """Home page: full-run results, the learning curve, and the live-demo form."""
    body = f"""
<h1>Smart Elevator: Q-learning</h1>
<p>A tabular Q-learning agent controls one elevator in a 5-floor building. It is compared with a
random policy and a rule-based policy on identical passenger arrivals.</p>
<h2>Full training run (20,000 episodes)</h2>
{table_html(FULL_RESULTS)}
<img src="/results/learning_curve.png" alt="Learning curve">
<h2>Try a live demo</h2>
<p>Trains a new agent right now. More episodes give a better agent but take longer
(up to {MAX_DEMO_EPISODES}).</p>
<form action="/demo" method="get">
  Episodes: <input type="number" name="episodes" min="10" max="{MAX_DEMO_EPISODES}"
  value="{DEFAULT_DEMO_EPISODES}"> <button type="submit">Train and compare</button>
</form>"""
    return page("Smart Elevator", body)


@app.route("/demo")
def demo():
    """Train a small agent for the requested number of episodes, then evaluate all three
    policies on the same seeds and show the comparison table."""
    n = request.args.get("episodes", default=DEFAULT_DEMO_EPISODES, type=int)
    n = max(10, min(n, MAX_DEMO_EPISODES))             # keep it between 10 and the maximum

    env = ElevatorEnv(seed=TRAIN_ENV_SEED)
    agent = QLearningAgent(env)
    train(agent, env, n, log_every=n + 1)              # log_every > n means no progress printing
    results = {
        "Random": evaluate(RandomPolicy(seed=RANDOM_POLICY_SEED), EVAL_EPISODES, EVAL_SEED),
        "Rule-based": evaluate(rule_based_policy, EVAL_EPISODES, EVAL_SEED),
        f"Q-learning ({n} ep.)": evaluate(agent, EVAL_EPISODES, EVAL_SEED),
    }
    body = (f"<h1>Live demo: {n} training episodes</h1>{table_html(results)}"
            "<p>With so little training the agent is usually worse than the rule-based policy. "
            "The full 20,000-episode run beats it.</p><p><a href='/'>Back</a></p>")
    return page("Live demo", body)


@app.route("/results/<path:name>")
def results_file(name):
    """Serve the saved learning-curve image and training log from the results/ folder."""
    return send_from_directory(os.path.join(HERE, "results"), name)


@app.route("/health")
def health():
    """Simple check that the server is up (Render can use this as a health check path)."""
    return "ok"


if __name__ == "__main__":
    app.run(debug=False)    # local use only; Render starts the app with gunicorn instead
