
"""Single PPO trainer shared by M1-M4 (clipped objective + GAE, truncation-aware)."""
import numpy as np
import torch


# Welford running std used to scale rewards identically for every model.
class RunningStd:
    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, x):
        for v in x:
            self.n += 1
            d = v - self.mean
            self.mean += d / self.n
            self.m2 += d * (v - self.mean)

    @property
    def std(self):
        return float(np.sqrt(self.m2 / max(self.n - 1, 1))) + 1e-8


# GAE with episode cuts: ends[t] marks a segment end; final_values[t] = V(s_final) (0 if terminal).
def compute_gae(rewards, values, ends, final_values, last_value, gamma, lam):
    T = len(rewards)
    adv, gae = np.zeros(T), 0.0
    for t in reversed(range(T)):
        if ends[t]:
            nv, gae = final_values[t], 0.0          # cut the recursion at the boundary
        else:
            nv = values[t + 1] if t + 1 < T else last_value
        delta = rewards[t] + gamma * nv - values[t]
        gae = delta + gamma * lam * (0.0 if ends[t] else gae)
        adv[t] = gae
    return adv, adv + np.asarray(values)           # advantages, value targets


class PPO:
    # p: PPOConfig-like object (gamma, clip, value_coef, entropy_coef, ...).
    def __init__(self, model, env, p, device, seed):
        self.m, self.env, self.p, self.dev = model.to(device), env, p, device
        self.rng = np.random.default_rng(seed)
        self.opt = torch.optim.Adam(model.parameters(), lr=p.lr)
        self.rs = RunningStd()
        self.obs, _ = env.reset()
        self.ep_return = 0.0  

    # Converts one numpy obs (W,D) to a (1,W,D) float tensor on the device.
    def _t(self, o):
        return torch.as_tensor(o, dtype=torch.float32, device=self.dev).unsqueeze(0)

    # Runs rollout_steps env steps, storing RAW actions and their old log-probs.
    def collect(self):
        B = {k: [] for k in ("obs", "raw", "logp", "v", "r", "end", "fv")}
        for _ in range(self.p.rollout_steps):
            raw, logp, v = self.m.act(self._t(self.obs), self.rng)
            nxt, r, terminated, truncated, info = self.env.step(raw)   # env thresholds internally
            fv = 0.0
            if terminated or truncated:
                if not terminated:
                    if not info.get("next_obs_valid", False):
                        raise RuntimeError("segment ends without a valid final observation")
                    fv = self.m.value(self._t(nxt))                     # bootstrap V(s_final)
            
            for k, x in zip(B, (self.obs, raw, logp, v, r, terminated or truncated, fv)):
                B[k].append(x)
            self.ep_return += r
            self.obs = self.env.reset()[0] if (terminated or truncated) else nxt 
                   
        last_v = self.m.value(self._t(self.obs))
        r = np.asarray(B["r"], dtype=np.float64)
        if self.p.normalize_reward:
            self.rs.update(r)
            r = r / self.rs.std
        adv, ret = compute_gae(r, B["v"], B["end"], B["fv"], last_v, self.p.gamma, self.p.gae_lambda)
        f = lambda x, dt=torch.float32: torch.as_tensor(np.asarray(x), dtype=dt, device=self.dev)
        return dict(obs=f(B["obs"]), raw=f(B["raw"], torch.float64), logp=f(B["logp"]),
                    adv=f(adv), ret=f(ret))

    # Several epochs of minibatch clipped-PPO updates on one rollout.
    def update(self, d):  
        
        n = len(d["adv"])
        stats = {}
        for _ in range(self.p.epochs):
            for idx in torch.randperm(n, device=self.dev).split(self.p.minibatch):
                adv = d["adv"][idx]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                logp, ent, v = self.m.evaluate(d["obs"][idx], d["raw"][idx])
                ratio = (logp - d["logp"][idx]).exp()
                pl = -torch.min(ratio * adv, ratio.clamp(1 - self.p.clip, 1 + self.p.clip) * adv).mean()
                vl = (v - d["ret"][idx]).pow(2).mean()
                loss = pl + self.p.value_coef * vl - self.p.entropy_coef * ent.mean()
               
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.m.parameters(), self.p.max_grad_norm)
               
                self.opt.step()
                stats = dict(pl=float(pl), vl=float(vl), ent=float(ent.mean()))
        return stats

    # Trains for total_steps env steps; returns the list of per-update stats.
    def train(self, total_steps):
        log, done = [], 0
        while done < total_steps:
            log.append(self.update(self.collect()))
            done += self.p.rollout_steps
        return log