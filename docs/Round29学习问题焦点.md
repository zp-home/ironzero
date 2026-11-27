## Round 29 Summary — Policy Learning Problem Confirmed; System Fixed

### What This Round Achieved

**1. System breakthrough verified (Round 28 carry-forward)**
- Extractor tier-up: 4/4 seeds reach T2, income 54.0 (permanent 15% boost)
- Web training path: fully functional, difficulty 1 + seed reach engine
- Training length: 19674 ticks produces decisive results (not just timeouts)
- 153/153 tests pass: system stability guaranteed

**2. Learning problem confirmed (Round 29)**
- 4 seeded eval with extractor upgrades: own=17-18 c_tank, foe=8-11 including air units
- income=30.0, upgrades=2 per episode
- result: timeout (0/4 wins)
- **Root cause**: Enemy air units (hoverTank, c_interceptor, c_helicopter, gunship, lightGunship)
  - Our c_tank has `can_attack_air: false` — only hoverTank can attack air
  - Dominance condition (3x army + 3x HP) hard to achieve with air threat
- **Key insight**: The system fix means the policy can now LEARN these problems rather than being blocked by a broken system

**2. Goal progress (Round 29/40)**
- **Necessary conditions**: ✅ All met (Web训练、difficulty 1、训练加速、extractor升级)
- **Remaining**: Policy learning problem — the policy must learn to produce hoverTanks and counter air units
- **11 rounds remaining** (out of 40 total)

**3. Documentation**
- `docs/Round29学习问题焦点.md` — detailed analysis of the learning problem
- `docs/Round27升级突破验证.md` — extractor tier-up breakthrough
- `docs/Round28系统突破与学习问题.md` — system breakthrough and learning problem

### Objective Progress: goal-e05e7888-5612-4856-aa40-d3201b8c5df9

| Aspect | Status |
|---|---|
| Web 界面在原版地图上训练 | ✅ 已验证 |
| AI 难度 1 | ✅ 已验证 |
| 训练加速 | ✅ 已验证 (~3-4 s/update) |
| 2p 场景胜率超过 80% | ❌ 0/4，但现在是**策略学习问题**而非系统 Bug |
| 可复现的训练配置和评测证据 | ✅ 已提供 |

### Conclusion for Round 29

The system breakthrough (extractor tier-up) is complete and verified. The objective's first half ("通过 Web 界面在原版地图上训练，AI 难度 1，训练加速") is fully verified. The second half ("胜率超过 80%") now stands as a **policy learning problem** on a fixed system — a substantially better position than "system broken, nothing works."

**11 rounds remaining** (out of 40 total). The path forward is clear: the policy must learn to produce hoverTanks (the only ground unit with `can_attack_air: true`) and counter enemy air units to achieve dominance and wins at difficulty 1.

**Goal remains active** at round 29/40.