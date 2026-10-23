"""
Tabular Q-learning for the Smart Elevator (ElevatorEnv).

HOW TO RUN (the environment file must be in the SAME folder as this file):
    python q_learning_elevator.py --test     # quick check: 50 episodes (about 1 second)
    python q_learning_elevator.py            # full run: 20,000 episodes (about 1-3 minutes)

In a notebook you can also call  main(["--test"])  or  main([]).
"""

import argparse
import os
import time

import numpy as np

# ----------------------------------------------------------------------
# Import the environment. The file is normally called elevator_env.py,
# but we also accept ElevatorEnv.py so the file name cannot break the import.
# ----------------------------------------------------------------------
try:
    from elevator_env import (ElevatorEnv, RandomPolicy, rule_based_policy,
                              evaluate, print_results_table)
except ModuleNotFoundError:
    try:
        from ElevatorEnv import (ElevatorEnv, RandomPolicy, rule_based_policy,
                                 evaluate, print_results_table)
    except ModuleNotFoundError:
        raise SystemExit(
            "\nCould not find the environment file.\n"
            "Save it as 'elevator_env.py' in the SAME folder as this file\n"
            "(in Colab: use the %writefile cell, or upload it), then run again.\n"
            f"Current folder: {os.getcwd()}\nFiles here: {os.listdir('.')}\n")

# ----------------------------------------------------------------------
# Fixed seeds, so results are reproducible
# ----------------------------------------------------------------------
TRAIN_ENV_SEED = 12345      # random passenger arrivals during training
AGENT_SEED = 1              # exploration and tie-breaking in the agent
RANDOM_POLICY_SEED = 123    # the Random baseline's own action choices
EVAL_SEED = 0               # evaluation episodes use seeds 0, 1, 2, ... (same for every policy)

# ----------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------
ALPHA = 0.1                 # learning rate
GAMMA = 0.95                # discount factor
EPS_START = 1.0             # epsilon at the start of training
EPS_END = 0.05              # epsilon after decay
FULL_EPISODES = 20000
TEST_EPISODES = 50
EVAL_EPISODES = 100


class QLearningAgent:
    """Tabular Q-learning agent: a NumPy Q-table plus epsilon-greedy action choice."""

    def __init__(self, env, alpha=ALPHA, gamma=GAMMA, eps_start=EPS_START,
                 eps_end=EPS_END, seed=AGENT_SEED):
        """Store the hyperparameters and create the Q-table: one row per encoded state
        (env.n_states rows), one column per action (env.N_ACTIONS columns), all zeros."""
        self.env = env                    # used only for env.encode_state()
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.q = np.zeros((env.n_states, env.N_ACTIONS))
        self.rng = np.random.default_rng(seed)   # agent's own random generator

    def epsilon(self, episode, n_episodes, decay_fraction=0.75):
        """Exploration rate for this episode. It decays smoothly (exponentially) from
        eps_start to eps_end over the first 75% of training, then stays at eps_end."""
        progress = min(1.0, episode / (decay_fraction * n_episodes))
        return self.eps_start * (self.eps_end / self.eps_start) ** progress

    def best_action(self, idx):
        """Return the action with the highest Q-value in table row `idx`.
        If several actions tie (e.g. a never-visited state), pick one of them at random."""
        row = self.q[idx]
        best = np.flatnonzero(row == row.max())
        if len(best) == 1:
            return int(best[0])
        return int(self.rng.choice(best))

    def choose_action(self, idx, epsilon):
        """Epsilon-greedy: with probability epsilon try a random action (explore),
        otherwise take the best known action (exploit)."""
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.env.N_ACTIONS))
        return self.best_action(idx)

    def update(self, idx, action, reward, next_idx, done):
        """The Q-learning update rule:
        Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a)).
        On the last step of an episode there is no next state, so the target is just the reward."""
        target = reward if done else reward + self.gamma * self.q[next_idx].max()
        self.q[idx, action] += self.alpha * (target - self.q[idx, action])

    def __call__(self, state):
        """Greedy policy (epsilon = 0): takes a state tuple, returns an action.
        This lets the agent be passed straight into evaluate()."""
        return self.best_action(self.env.encode_state(state))


# ======================================================================
def check_environment(env):
    """Run one step and verify the environment behaves the way this file expects.
    If something is wrong, you get a clear message instead of a confusing crash later."""
    state = env.reset(seed=0)
    assert isinstance(state, tuple) and len(state) == 4, "reset() must return (floor, hall, cabin, tod)"
    idx = env.encode_state(state)
    assert 0 <= idx < env.n_states, "encode_state() must give an index inside the Q-table"
    result = env.step(ElevatorEnv.STAY)
    assert len(result) == 4, "step() must return (state, reward, done, info)"
    _, _, _, info = result
    for key in ("delivered", "moved", "waits"):
        assert key in info, f"step() info is missing '{key}'"
    env.reset(seed=0)   # leave the environment clean
    print("Environment check passed.")


def train(agent, env, n_episodes, log_every):
    """Train the agent for n_episodes. Returns a dict of per-episode NumPy arrays:
    total_reward, avg_wait (hall wait per passenger, in steps) and moves."""
    total_reward = np.zeros(n_episodes)
    avg_wait = np.zeros(n_episodes)
    moves = np.zeros(n_episodes, dtype=int)
    start = time.time()

    for ep in range(n_episodes):
        eps = agent.epsilon(ep, n_episodes)
        state = env.reset()                       # no seed: env's random stream just continues
        idx = env.encode_state(state)
        done = False
        ep_reward, ep_moves, waits = 0.0, 0, []

        while not done:                           # one episode = 200 steps
            action = agent.choose_action(idx, eps)
            next_state, reward, done, info = env.step(action)
            next_idx = env.encode_state(next_state)
            agent.update(idx, action, reward, next_idx, done)
            idx = next_idx
            ep_reward += reward
            ep_moves += int(info["moved"])
            waits.extend(info["waits"])           # waits of passengers who boarded this step

        waits.extend(env.waiting_times_now())     # passengers never picked up count too
        total_reward[ep] = ep_reward
        avg_wait[ep] = np.mean(waits) if waits else 0.0
        moves[ep] = ep_moves

        if (ep + 1) % log_every == 0:
            r = slice(ep + 1 - log_every, ep + 1)  # average over the last log_every episodes
            print(f"episode {ep + 1:6d} | eps {eps:.3f} | reward {total_reward[r].mean():8.1f} | "
                  f"avg wait {avg_wait[r].mean():6.2f} | moves {moves[r].mean():6.1f} | "
                  f"{time.time() - start:5.0f}s")

    return {"total_reward": total_reward, "avg_wait": avg_wait, "moves": moves}


def moving_average(x, window):
    """Smooth a 1-D array with a moving average (result has len(x) - window + 1 points)."""
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_learning_curve(logs, rule_reward, path, window):
    """Save a learning-curve PNG: faint raw rewards, bold moving average, and a dashed
    horizontal line at the rule-based policy's average reward."""
    import matplotlib.pyplot as plt

    rewards = logs["total_reward"]
    smooth = moving_average(rewards, window)
    x = np.arange(window, len(rewards) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(np.arange(1, len(rewards) + 1), rewards, color="tab:blue", alpha=0.12,
            label="Episode reward")
    ax.plot(x, smooth, color="tab:blue", linewidth=2,
            label=f"Q-learning ({window}-episode moving average)")
    ax.axhline(rule_reward, color="tab:red", linestyle="--",
               label=f"Rule-based ({rule_reward:.1f})")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Total reward per episode")
    ax.set_title("Q-learning elevator: learning curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_log_csv(logs, path):
    """Write the per-episode log to a CSV file: episode, total_reward, avg_wait, moves."""
    n = len(logs["total_reward"])
    data = np.column_stack([np.arange(1, n + 1), logs["total_reward"],
                            logs["avg_wait"], logs["moves"]])
    np.savetxt(path, data, delimiter=",", header="episode,total_reward,avg_wait,moves",
               comments="", fmt=["%d", "%.2f", "%.3f", "%d"])


# ======================================================================
def main(argv=None):
    """Check the environment, train, save the CSV and PNG, then print the comparison table.
    argv=None reads the command line; in a notebook pass a list, e.g. main(["--test"])."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help=f"quick test mode: {TEST_EPISODES} episodes")
    parser.add_argument("--episodes", type=int, default=None,
                        help=f"number of training episodes (default {FULL_EPISODES})")
    parser.add_argument("--out", default=".", help="folder for the CSV and PNG files")
    args, _ = parser.parse_known_args(argv)   # ignores extra arguments added by notebooks

    n_episodes = args.episodes or (TEST_EPISODES if args.test else FULL_EPISODES)
    prefix = "test_" if args.test else ""     # test runs do not overwrite the full-run files
    os.makedirs(args.out, exist_ok=True)
    log_every = 10 if n_episodes < 1000 else 1000
    window = min(200, max(1, n_episodes // 5))  # 200 for the full run, smaller for short runs

    train_env = ElevatorEnv(seed=TRAIN_ENV_SEED)
    check_environment(train_env)

    # Rule-based baseline first, so the plot can draw its reward as a horizontal line.
    rule_result = evaluate(rule_based_policy, EVAL_EPISODES, EVAL_SEED)

    agent = QLearningAgent(train_env)
    print(f"\nTraining for {n_episodes} episodes...\n")
    logs = train(agent, train_env, n_episodes, log_every)

    csv_path = os.path.join(args.out, f"{prefix}training_log.csv")
    png_path = os.path.join(args.out, f"{prefix}learning_curve.png")
    save_log_csv(logs, csv_path)
    plot_learning_curve(logs, rule_result["avg_reward"], png_path, window)
    print(f"\nSaved {csv_path} and {png_path}")

    # Greedy evaluation (epsilon = 0) with the SAME seeds as both baselines.
    results = {
        "Random": evaluate(RandomPolicy(seed=RANDOM_POLICY_SEED), EVAL_EPISODES, EVAL_SEED),
        "Rule-based": rule_result,
        "Q-learning": evaluate(agent, EVAL_EPISODES, EVAL_SEED),
    }
    print(f"\nComparison over {EVAL_EPISODES} episodes (same arrival seeds for every policy)\n")
    print_results_table(results)
    if args.test:
        print("\nTest mode finished OK. The Q-learning row is weak because it only trained "
              f"for {n_episodes} episodes. Run without --test for the real result.")


if __name__ == "__main__":
    main()
