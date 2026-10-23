"""
ElevatorEnv: a small elevator environment for tabular Q-learning (NumPy only),
plus two baseline policies and an evaluation function.

Floors are numbered 0..4 in code. Floor 0 is the lobby, i.e. "floor 1"
in everyday language.

State  = (elevator_floor, hall_calls, cabin_dests, time_bucket)
    elevator_floor : int 0..4
    hall_calls     : int 0..31, 5 bits. Bit f is 1 if someone waits on floor f.
    cabin_dests    : int 0..31, 5 bits. Bit f is 1 if a rider inside wants floor f.
    time_bucket    : int 0..3  (0 morning, 1 midday, 2 evening, 3 night)

Actions: 0 = UP, 1 = DOWN, 2 = STAY_SERVE (open doors: unload, then load)
"""

import numpy as np


class ElevatorEnv:
    # ----- constants -----
    N_FLOORS = 5
    UP, DOWN, STAY = 0, 1, 2
    N_ACTIONS = 3
    ACTION_NAMES = ["UP", "DOWN", "STAY"]

    # Chance that one new passenger appears in a step, per time bucket
    # (morning, midday, evening, night).
    ARRIVAL_PROB = [0.30, 0.20, 0.30, 0.05]
    # In morning/evening, how often a passenger follows the "rush" pattern.
    RUSH_FRACTION = 0.8

    # ----- rewards -----
    R_WAIT = -1.0      # per waiting passenger, per step
    R_MOVE = -0.5      # per floor moved
    R_DELIVER = 10.0   # per passenger delivered
    R_USELESS = -2.0   # useless STAY, or hitting a wall

    def __init__(self, seed=None, episode_length=200):
        """Create the environment. `seed` makes the random arrivals repeatable."""
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)
        self.n_states = self.N_FLOORS * 32 * 32 * 4  # size of a Q-table's state axis
        self.reset()

    # ------------------------------------------------------------------
    def reset(self, seed=None):
        """Start a new episode: empty building, elevator in the lobby.
        Pass `seed` to reseed the random generator. Returns the first state."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.floor = 0
        self.t = 0
        # waiting[f] is a list of (destination, arrival_step) for passengers waiting on floor f
        self.waiting = [[] for _ in range(self.N_FLOORS)]
        # cabin is a list of destinations of passengers riding the elevator
        self.cabin = []
        self.done = False
        return self._get_state()

    # ------------------------------------------------------------------
    def step(self, action):
        """Apply one action (0 UP, 1 DOWN, 2 STAY). Returns (state, reward, done, info).
        info has: delivered (int), moved (bool), waits (list of hall waiting times, in steps,
        of passengers who boarded this step), waiting (int), in_cabin (int)."""
        if self.done:
            raise RuntimeError("Episode is over, call reset() first.")
        if action not in (self.UP, self.DOWN, self.STAY):
            raise ValueError(f"Invalid action {action}")

        reward = 0.0
        delivered = 0
        moved = False
        waits = []

        if action == self.UP:
            if self.floor == self.N_FLOORS - 1:
                reward += self.R_USELESS          # hit the top wall, no movement
            else:
                self.floor += 1
                moved = True
                reward += self.R_MOVE
        elif action == self.DOWN:
            if self.floor == 0:
                reward += self.R_USELESS          # hit the bottom wall, no movement
            else:
                self.floor -= 1
                moved = True
                reward += self.R_MOVE
        else:  # STAY_SERVE
            delivered, waits = self._serve_current_floor()
            reward += self.R_DELIVER * delivered
            if delivered == 0 and len(waits) == 0:
                reward += self.R_USELESS          # doors opened for nobody

        # New passengers may appear after the action is processed.
        self._spawn_passenger()
        self.t += 1

        # Penalty for everyone still waiting in the halls at the end of the step.
        n_waiting = sum(len(w) for w in self.waiting)
        reward += self.R_WAIT * n_waiting

        self.done = self.t >= self.episode_length
        info = {"delivered": delivered, "moved": moved, "waits": waits,
                "waiting": n_waiting, "in_cabin": len(self.cabin)}
        return self._get_state(), reward, self.done, info

    # ------------------------------------------------------------------
    def _serve_current_floor(self):
        """Unload riders whose destination is here, then load everyone waiting here.
        Returns (number delivered, list of waiting times of the passengers who boarded)."""
        delivered = sum(1 for d in self.cabin if d == self.floor)
        self.cabin = [d for d in self.cabin if d != self.floor]
        waits = []
        for dest, arrival_step in self.waiting[self.floor]:
            self.cabin.append(dest)
            waits.append(self.t - arrival_step)
        self.waiting[self.floor] = []
        return delivered, waits

    # ------------------------------------------------------------------
    def waiting_times_now(self):
        """Return how long (in steps) each passenger still waiting in a hall has waited so far."""
        return [self.t - arrival for hall in self.waiting for _, arrival in hall]

    # ------------------------------------------------------------------
    def _time_bucket(self):
        """Map the current step to a time-of-day bucket 0..3 (equal quarters of the episode)."""
        return min(3, self.t * 4 // self.episode_length)

    # ------------------------------------------------------------------
    def _random_trip(self):
        """Pick a uniformly random origin and a different destination."""
        origin = int(self.rng.integers(self.N_FLOORS))
        dest = int(self.rng.integers(self.N_FLOORS - 1))
        if dest >= origin:       # skip over the origin so dest != origin
            dest += 1
        return origin, dest

    # ------------------------------------------------------------------
    def _spawn_passenger(self):
        """With a time-dependent probability, add one passenger to a hall.
        Morning: mostly lobby -> upper floors. Evening: mostly upper floors -> lobby.
        Midday and night: random trips."""
        bucket = self._time_bucket()
        if self.rng.random() >= self.ARRIVAL_PROB[bucket]:
            return
        rush = self.rng.random() < self.RUSH_FRACTION
        if bucket == 0 and rush:
            origin, dest = 0, int(self.rng.integers(1, self.N_FLOORS))
        elif bucket == 2 and rush:
            origin, dest = int(self.rng.integers(1, self.N_FLOORS)), 0
        else:
            origin, dest = self._random_trip()
        self.waiting[origin].append((dest, self.t))

    # ------------------------------------------------------------------
    def _get_state(self):
        """Build the state tuple (floor, hall_calls bits, cabin bits, time bucket)."""
        hall = sum(1 << f for f in range(self.N_FLOORS) if self.waiting[f])
        cabin = sum(1 << f for f in set(self.cabin))
        return (self.floor, hall, cabin, self._time_bucket())

    # ------------------------------------------------------------------
    def encode_state(self, state):
        """Turn a state tuple into one integer 0..n_states-1, handy as a Q-table row index."""
        floor, hall, cabin, tod = state
        return ((floor * 32 + hall) * 32 + cabin) * 4 + tod

    # ------------------------------------------------------------------
    @staticmethod
    def format_state(state):
        """Readable string for a state, with the two 5-bit fields shown in binary."""
        floor, hall, cabin, tod = state
        return f"floor={floor} hall={hall:05b} cabin={cabin:05b} tod={tod}"


# ======================================================================
# Policies. A policy is any callable that takes a state tuple and returns an action.
# ======================================================================
class RandomPolicy:
    """Picks UP, DOWN or STAY uniformly at random."""

    def __init__(self, seed=None):
        """Give the policy its own random generator so it never disturbs the environment's."""
        self.rng = np.random.default_rng(seed)

    def __call__(self, state):
        """Return a random action; the state is ignored."""
        return int(self.rng.integers(ElevatorEnv.N_ACTIONS))


def rule_based_policy(state):
    """Hand-written baseline.
    1) If there is a hall call or a cabin destination on the current floor: STAY (serve it).
    2) Otherwise move one floor toward the nearest call. Cabin destinations count as calls
       too, otherwise riders could be stuck inside with nobody pressing a hall button.
       Ties between equally near floors go to the lower floor.
    3) If there are no calls at all: STAY."""
    floor, hall, cabin, _tod = state
    requests = hall | cabin                      # combine both 5-bit masks
    if (requests >> floor) & 1:                  # something to do right here
        return ElevatorEnv.STAY
    targets = [f for f in range(ElevatorEnv.N_FLOORS) if (requests >> f) & 1]
    if not targets:                              # nothing to do anywhere
        return ElevatorEnv.STAY
    nearest = min(targets, key=lambda f: (abs(f - floor), f))
    return ElevatorEnv.UP if nearest > floor else ElevatorEnv.DOWN


# ======================================================================
def evaluate(policy, n_episodes=100, seed=0):
    """Run `policy` for `n_episodes` episodes and return a dict of results.

    Episode i always uses env.reset(seed=seed + i), so every policy sees exactly the
    same passenger arrivals and the comparison is fair.

    Returned keys:
      avg_reward  : mean total reward per episode
      avg_wait    : mean hall waiting time per passenger, in steps
      max_wait    : longest hall waiting time of any passenger, in steps
      total_moves : floors moved, summed over all episodes
      delivered   : passengers delivered, summed over all episodes

    Waiting time = steps from arriving in the hall until boarding. Passengers still
    waiting when an episode ends are counted with the time they have waited so far,
    so a policy cannot look good by ignoring someone forever."""
    env = ElevatorEnv()
    totals = []
    all_waits = []
    total_moves = 0
    total_delivered = 0

    for ep in range(n_episodes):
        state = env.reset(seed=seed + ep)
        done = False
        ep_reward = 0.0
        while not done:
            action = policy(state)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            total_moves += int(info["moved"])
            total_delivered += info["delivered"]
            all_waits.extend(info["waits"])
        all_waits.extend(env.waiting_times_now())   # count the ones never picked up
        totals.append(ep_reward)

    return {
        "avg_reward": float(np.mean(totals)),
        "avg_wait": float(np.mean(all_waits)) if all_waits else 0.0,
        "max_wait": int(max(all_waits)) if all_waits else 0,
        "total_moves": total_moves,
        "delivered": total_delivered,
    }


def print_results_table(results):
    """Print a dict {policy name: evaluate() result} as an aligned text table."""
    header = (f"{'Policy':<12} {'Avg reward':>11} {'Avg wait':>9} {'Max wait':>9} "
              f"{'Total moves':>12} {'Delivered':>10}")
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<12} {r['avg_reward']:>11.1f} {r['avg_wait']:>9.2f} {r['max_wait']:>9d} "
              f"{r['total_moves']:>12d} {r['delivered']:>10d}")


# ======================================================================
# Tests
# ======================================================================
def test_random_episodes(n_episodes=5, steps_to_print=10, seed=42):
    """Run random-action episodes, print the first few steps of each,
    and check that every episode lasts exactly 200 steps."""
    env = ElevatorEnv(seed=seed)
    policy = RandomPolicy(seed + 1)

    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0.0
        steps = 0
        print(f"\n=== Episode {ep + 1} ===")
        done = False
        while not done:
            action = policy(state)
            next_state, reward, done, info = env.step(action)
            if steps < steps_to_print:
                print(f"step {steps + 1:2d} | state: {ElevatorEnv.format_state(state)} "
                      f"| action: {ElevatorEnv.ACTION_NAMES[action]:<4} | reward: {reward:6.1f}")
            total_reward += reward
            state = next_state
            steps += 1
        assert steps == 200, "Episode should last exactly 200 steps"
        print(f"Episode {ep + 1} finished: {steps} steps, total reward = {total_reward:.1f}")

    print("\nAll tests passed.")


def test_policy_comparison(n_episodes=100, seed=0):
    """Evaluate both policies on identical arrivals, print the table, and check
    that the rule-based policy beats random on average reward."""
    results = {
        "Random": evaluate(RandomPolicy(seed=123), n_episodes, seed),
        "Rule-based": evaluate(rule_based_policy, n_episodes, seed),
    }
    print(f"\nComparison over {n_episodes} episodes (same arrival seeds for every policy)\n")
    print_results_table(results)
    assert results["Rule-based"]["avg_reward"] > results["Random"]["avg_reward"]
    print("\nRule-based beats random: OK")


if __name__ == "__main__":
    test_random_episodes()
    test_policy_comparison()
