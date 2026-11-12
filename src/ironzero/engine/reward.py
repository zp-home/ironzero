# SPDX-License-Identifier: Apache-2.0
"""RewardMixin: potential-based shaping plus economy/terminal components.

``_reward_after_window()`` is called once per decision window from
``MicroDuelEngine.advance()``; it also resets the per-window economic counters.
The reward decomposition is stored in ``self.latest_reward_components`` so the
env can expose it without recomputation.
"""

from __future__ import annotations


class RewardMixin:
    """Potential-based reward shaping and its component breakdown."""

    def _potential(self) -> float:
        cfg = self.config
        total = float(cfg.units_per_team * cfg.unit_hp)
        own = sum(u.hp for u in self.alive_units(0)) / total
        enemy = sum(u.hp for u in self.alive_units(1)) / total
        return own - enemy

    def _reward_after_window(self) -> float:
        cfg = self.config
        phi = self._potential()
        potential_diff = cfg.potential_coef * (phi - self._phi_prev)
        self._phi_prev = phi

        reward = potential_diff
        ev = self.economic_events.get(0, {}) if hasattr(self, "economic_events") else {}
        econ_reward = 0.0
        factories_reward = 0.0
        prod_reward = 0.0
        building_loss_penalty = 0.0

        if cfg.enable_economy:
            econ_reward = (
                getattr(cfg, "extractor_reward", 0.5) * ev.get("extractor_built", 0)
                + getattr(cfg, "extractor_t2_reward", 0.8) * ev.get("extractor_upgraded", 0)
            )
            factories_reward = getattr(cfg, "factory_reward", 0.6) * ev.get("factory_built", 0)
            prod_reward = getattr(cfg, "production_reward", 0.4) * ev.get("unit_produced", 0)
            building_loss_penalty = getattr(cfg, "building_loss_penalty", 1.0) * ev.get("building_destroyed", 0)
            reward += (econ_reward + factories_reward + prod_reward - building_loss_penalty)

            # Idle funds penalty: capital sitting in the bank instead of being
            # converted into production capacity.
            funds = self.funds.get(0, 0.0)
            threshold = float(getattr(cfg, "idle_funds_threshold", 800.0))
            if funds > threshold:
                reward -= float(getattr(cfg, "idle_funds_penalty_coef", 0.0001)) * (funds - threshold)

        terminal_reward = 0.0
        if self.done:
            if self.result == "victory":
                terminal_reward = cfg.win_reward
                reward += terminal_reward
            elif self.result == "defeat":
                terminal_reward = cfg.loss_reward
                reward += terminal_reward

        self.latest_reward_components = {
            "potential": float(potential_diff),
            "economy": float(econ_reward),
            "factories": float(factories_reward),
            "production": float(prod_reward),
            "building_loss": float(-building_loss_penalty),
            "terminal": float(terminal_reward),
        }

        if hasattr(self, "economic_events"):
            for team in range(cfg.teams):
                self.economic_events[team] = {
                    "extractor_built": 0,
                    "extractor_upgraded": 0,
                    "factory_built": 0,
                    "unit_produced": 0,
                    "building_destroyed": 0,
                }

        return float(reward)
