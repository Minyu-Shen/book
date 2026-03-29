#!/usr/bin/env python3
"""
Train PPO on the ticket pricing MDP and export data for web visualization.

Usage:  python train_ppo_pricing.py
Output: ppo_data.js (same directory)
"""

import torch, torch.nn as nn, numpy as np, json, os, time, math


# ===================== Environment =====================

class TicketPricingEnv:
    PRICES  = [200, 400, 600, 800]
    LAMBDAS = [8.0 * math.exp(-p / 400.0) for p in PRICES]

    def __init__(self, total_seats=50, total_days=30):
        self.total_seats = total_seats
        self.total_days  = total_days

    def reset(self):
        self.seats     = self.total_seats
        self.days_left = self.total_days
        return self._obs()

    def _obs(self):
        return (self.seats / self.total_seats, self.days_left / self.total_days)

    def step(self, a):
        price  = self.PRICES[a]
        demand = np.random.poisson(self.LAMBDAS[a])
        sold   = min(demand, self.seats)
        reward = price * sold
        self.seats     -= sold
        self.days_left -= 1
        done = (self.days_left <= 0) or (self.seats <= 0)
        return self._obs(), reward, done


# ===================== Actor-Critic =====================

class ActorCritic(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.pi = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 4),
        )
        self.v = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.pi(x), self.v(x).squeeze(-1)


# ===================== GAE =====================

def compute_gae(rewards, values, dones, gamma, lam):
    n = len(rewards)
    adv = [0.0] * n
    g = 0.0
    for t in range(n - 1, -1, -1):
        nv = 0.0 if (t == n - 1 or dones[t]) else values[t + 1]
        delta = rewards[t] + gamma * nv - values[t]
        g = delta if dones[t] else delta + gamma * lam * g
        adv[t] = g
    return adv


# ===================== DP Optimal =====================

def dp_optimal_undiscounted():
    """Compute optimal expected total revenue (gamma=1) via backward induction."""
    prices = TicketPricingEnv.PRICES
    lams   = TicketPricingEnv.LAMBDAS

    V   = np.zeros((51, 31))   # V[seats][days_left]
    pol = np.zeros((51, 31), dtype=int)

    for d in range(1, 31):
        for s in range(1, 51):
            best_val = -1e18
            for a in range(4):
                ev, cum_prob = 0.0, 0.0
                for k in range(s):
                    pmf = math.exp(-lams[a]) * (lams[a] ** k) / math.factorial(k)
                    ev += pmf * (prices[a] * k + V[s - k][d - 1])
                    cum_prob += pmf
                # demand >= s: sell all remaining seats
                ev += (1.0 - cum_prob) * (prices[a] * s + V[0][d - 1])
                if ev > best_val:
                    best_val = ev
                    pol[s][d] = a
            V[s][d] = best_val

    return V[50][30], V, pol


# ===================== Training =====================

TRACK_STATES = torch.tensor([
    [40 / 50, 25 / 30],  # 40 seats, 25 days
    [ 5 / 50, 25 / 30],  # 5 seats,  25 days
    [40 / 50,  3 / 30],  # 40 seats, 3 days
    [ 5 / 50,  3 / 30],  # 5 seats,  3 days
], dtype=torch.float32)

REWARD_SCALE = 1000.0   # normalize rewards for stable critic training


def train_ppo(
    clip_eps=0.2,     # None = no clipping
    K=3,              # update passes per batch
    gae_lambda=0.95,  # 0 = pure 1-step TD
    n_epochs=200,
    batch_size=32,    # enough episodes for stable estimates
    gamma=0.99,
    lr=1e-3,          # large enough that clipping actually activates
    seed=42,
    save_heatmap=False,
    tag="",
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env   = TicketPricingEnv()
    model = ActorCritic()
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    use_clip = clip_eps is not None

    rew_hist = []
    pol_hist = [[] for _ in range(4)]
    obs_buf  = torch.zeros(1, 2)

    for epoch in range(n_epochs):
        # ---- collect episodes ----
        S, A, R, D, LP, V = [], [], [], [], [], []
        total_ep_reward = 0.0

        for _ in range(batch_size):
            o = env.reset()
            ep_r = 0.0
            while True:
                obs_buf[0, 0], obs_buf[0, 1] = o
                with torch.no_grad():
                    logits, v = model(obs_buf)
                dist = torch.distributions.Categorical(logits=logits[0])
                a = dist.sample()

                S.append(o)
                A.append(a.item())
                LP.append(dist.log_prob(a).item())
                V.append(v.item())

                o2, r, d = env.step(a.item())
                R.append(r / REWARD_SCALE)
                D.append(d)
                ep_r += r
                o = o2
                if d:
                    break
            total_ep_reward += ep_r

        rew_hist.append(total_ep_reward / batch_size)

        # policy snapshots at tracked states
        with torch.no_grad():
            probs = torch.softmax(model.pi(TRACK_STATES), dim=-1)
        for i in range(4):
            pol_hist[i].append([round(x, 4) for x in probs[i].tolist()])

        # ---- compute GAE ----
        advs  = compute_gae(R, V, D, gamma, gae_lambda)
        s_t   = torch.tensor(S)
        a_t   = torch.tensor(A, dtype=torch.long)
        olp_t = torch.tensor(LP)
        adv_t = torch.tensor(advs, dtype=torch.float32)
        ret_t = adv_t + torch.tensor(V, dtype=torch.float32)

        if adv_t.std() > 1e-8:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # ---- K update passes ----
        for _ in range(K):
            logits, v = model(s_t)
            dist  = torch.distributions.Categorical(logits=logits)
            nlp   = dist.log_prob(a_t)
            ratio = torch.exp(nlp - olp_t)

            if use_clip:
                s1 = ratio * adv_t
                s2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t
                pi_loss = -torch.min(s1, s2).mean()
            else:
                ratio = torch.clamp(ratio, 0.01, 100.0)  # numerical safety
                pi_loss = -(ratio * adv_t).mean()

            v_loss = (ret_t - v).pow(2).mean()
            loss   = pi_loss + 0.5 * v_loss - 0.01 * dist.entropy().mean()

            if torch.isnan(loss) or torch.isinf(loss):
                break
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

        if (epoch + 1) % 50 == 0:
            print(f"  [{tag:>10s}] epoch {epoch+1:3d}/{n_epochs}  "
                  f"reward = {rew_hist[-1]:,.0f}")

    # ---- heatmap: converged policy over entire state space ----
    hm = None
    if save_heatmap:
        hm = []
        for seats in range(1, 51):
            row = []
            for days in range(1, 31):
                inp = torch.tensor([[seats / 50, days / 30]])
                with torch.no_grad():
                    p = torch.softmax(model.pi(inp), dim=-1)[0]
                row.append([round(x, 4) for x in p.tolist()])
            hm.append(row)

    return {
        "rewards": [round(x, 1) for x in rew_hist],
        "policy":  pol_hist,
        "heatmap": hm,
    }


# ===================== Utilities =====================

def smooth(arr, window=7):
    """Symmetric moving-average smoothing."""
    out = []
    hw = window // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - hw), min(len(arr), i + hw + 1)
        out.append(round(sum(arr[lo:hi]) / (hi - lo), 1))
    return out


def multi_seed_run(cfg, seeds, n_epochs=200):
    """Run config across multiple seeds, return mean/lo/hi curves + best policy/heatmap."""
    all_rewards = []
    best_result = None
    best_final  = -1e18

    for s in seeds:
        r = train_ppo(**cfg, n_epochs=n_epochs, seed=s, tag=f"s{s}")
        all_rewards.append(r["rewards"])
        if r["rewards"][-1] > best_final:
            best_final = r["rewards"][-1]
            best_result = r

    # compute per-epoch stats
    arr = np.array(all_rewards)  # (n_seeds, n_epochs)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    lo   = mean - std
    hi   = mean + std

    return {
        "mean": smooth(mean.tolist()),
        "lo":   smooth(lo.tolist()),
        "hi":   smooth(hi.tolist()),
        "policy":  best_result["policy"],
        "heatmap": best_result["heatmap"],
    }


# ===================== Main =====================

SEEDS = [42, 43, 44, 45, 46]

def main():
    t0 = time.time()

    # DP reference
    dp_val, _, _ = dp_optimal_undiscounted()
    print(f"DP optimal (undiscounted): {dp_val:,.0f}")

    # All training configurations  (clip_eps, K, gae_lambda)
    configs = {
        # Demo 1 = Demo2-K3 = Demo3-eps0.2 = Ablation-111
        "full_ppo": dict(clip_eps=0.2,  K=3,  gae_lambda=0.95, save_heatmap=True),
        # Demo 2 — K comparison
        "K1":       dict(clip_eps=0.2,  K=1,  gae_lambda=0.95),
        "K10":      dict(clip_eps=0.2,  K=10, gae_lambda=0.95),
        # Demo 3 — epsilon comparison
        "eps01":    dict(clip_eps=0.1,  K=3,  gae_lambda=0.95),
        "eps04":    dict(clip_eps=0.4,  K=3,  gae_lambda=0.95),
        # Ablation 8 combos [clip, multiK, GAE]
        "abl_000":  dict(clip_eps=None, K=1,  gae_lambda=0.0),    # baseline
        "abl_100":  dict(clip_eps=0.2,  K=1,  gae_lambda=0.0),    # +clip
        "abl_010":  dict(clip_eps=None, K=3,  gae_lambda=0.0),    # +multiK
        "abl_001":  dict(clip_eps=None, K=1,  gae_lambda=0.95),   # +GAE
        "abl_110":  dict(clip_eps=0.2,  K=3,  gae_lambda=0.0),    # +clip+multiK
        # abl_101 = K1,  abl_111 = full_ppo
        "abl_011":  dict(clip_eps=None, K=3,  gae_lambda=0.95),   # +multiK+GAE
    }

    results = {}
    for name, cfg in configs.items():
        print(f"\n{'='*40}\nConfig: {name}  ({len(SEEDS)} seeds)")
        results[name] = multi_seed_run(cfg, SEEDS)

    def curve(name):
        """Return {mean, lo, hi} for a config."""
        r = results[name]
        return {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"]}

    # ---- assemble output ----
    output = {
        "n_epochs":   200,
        "dp_optimal": round(dp_val, 1),
        "demo1": {
            "rewards": results["full_ppo"]["mean"],
            "policy":  results["full_ppo"]["policy"],
        },
        "demo2": {
            "K1":  curve("K1"),
            "K3":  curve("full_ppo"),
            "K10": curve("K10"),
        },
        "demo3": {
            "e01": curve("eps01"),
            "e02": curve("full_ppo"),
            "e04": curve("eps04"),
        },
        "demo4": [
            curve("abl_000"),                # 000 baseline
            curve("abl_100"),                # 100 +clip
            curve("abl_010"),                # 010 +multiK
            curve("abl_001"),                # 001 +GAE
            curve("abl_110"),                # 110 +clip+multiK
            curve("K1"),                      # 101 +clip+GAE (= K1)
            curve("abl_011"),                # 011 +multiK+GAE
            curve("full_ppo"),               # 111 full PPO
        ],
        "heatmap": results["full_ppo"]["heatmap"],
    }

    # ---- write JS data file ----
    out_dir  = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "ppo_data.js")
    with open(out_path, "w") as f:
        f.write("// Auto-generated by train_ppo_pricing.py — do not edit manually\n")
        f.write("var PPO_DATA = ")
        json.dump(output, f, separators=(",", ":"))
        f.write(";\n")

    elapsed = time.time() - t0
    fsize   = os.path.getsize(out_path) / 1024
    print(f"\n{'='*50}")
    print(f"Done in {elapsed:.1f}s  ({len(SEEDS)} seeds × {len(configs)} configs)")
    print(f"Output: {out_path}  ({fsize:.1f} KB)")
    print(f"DP optimal: {dp_val:,.0f}")
    m = results["full_ppo"]["mean"]
    print(f"PPO mean final reward: {m[-1]:,.0f}")


if __name__ == "__main__":
    main()
