# Round 27 Summary — Extractor Tier-Up Breakthrough Verified

**What was achieved this round:**
- The extractor tier-up system is now proven working end-to-end
- 4/4 curriculum seeds reach extractorT2 (tech_level 2) with `max_upgrades=4`
- Income rises from 26.0 → 30.0 when upgrades complete
- 153/153 tests pass

**Root cause fixed (Round 26, now verified):**
- Stock map extractors are `CustomUnit` not `Factory`
- `instanceof Factory` guard was always false → fallback path silently succeeded with no queue
- Added `CustomUnit` detection + `getUpgradeActions()` + targeted 4-param `executeSpecialAction`
- Same dispatch no longer runs production, preventing 1400-credit upgrade from being immediately outbid

**Verified results (scripts/probe_upgrade_curriculum.py):**
```
max_upgrades=0:  T2-present 0/4   tiers all extractorT1
max_upgrades=4:  T2-present 4/4   tiers include extractorT2, tech_level 2
peak_income=[54.0, 54.0, 54.0, 54.0]   (was oscillating 0-58 before, now stable)
issued=[2, 2, 2, 2]                     (2 upgrade orders per episode)
```

**Objective progress (Round 27/40):**
- ✅ Necessary condition met: extractor upgrades now complete, policy can learn economic benefit
- ❓ Sufficiency unclear: difficulty 1 still 0/4 win rate even with T2 (structural: x1.4 income + T2 air units + our tanks can't attack air)
- The fix eliminates the "upgrade was never completed" excuse; policy now has a chance to learn

**Next technical steps (if pursuing):**
1. Verify policy actually extracts income benefit in training (previously impossible)
2. Explore hoverTank usage (only ground unit with `can_attack_air: true`)
3. Consider reward shaping: "enemy building damaged" term identified as missing gradient in earlier rounds
4. May need config adjustments (army_cap, extractor count) beyond just the upgrade fix

**Current state**: Goal active, 27/40 rounds completed. The extractor upgrade breakthrough is the single most important structural fix since Round 18 (factory stall) and Round 19 (upgrade price correction). Without it, the policy could never benefit from the economy; now it can.