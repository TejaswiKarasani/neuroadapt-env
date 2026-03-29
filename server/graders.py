"""Deterministic trajectory graders for all NeuroAdapt tasks."""

try:
    from ..models import GraderResult, State
except ImportError:  # pragma: no cover - local script fallback
    from models import GraderResult, State


class Grader:

    def grade(self, task_id: str, state: State) -> GraderResult:
        if task_id == "easy":
            return self._grade_easy(state)
        elif task_id == "medium":
            return self._grade_medium(state)
        elif task_id == "hard":
            return self._grade_hard(state)
        return GraderResult(task_id=task_id, score=0.0, breakdown={}, passed=False)

    # ── Task 1: Profile Matching (Easy, 4 steps) ─────────────────────── #
    def _grade_easy(self, state: State) -> GraderResult:
        """
        Objective: Keep engagement high and stress low using the profile hint.
        Trajectory-only — no hidden state.

        Components:
          - engagement_score:  steps with engagement >= 0.55
          - low_stress_score:  steps with stress <= 0.50
          - correct_rate:      correct answers
          - variety_score:     used 2+ different content_types (calibrated for 4 steps)
        """
        traj = state.trajectory
        if not traj:
            return GraderResult(task_id="easy", score=0.0, breakdown={}, passed=False)

        n = len(traj)
        engagement_score = sum(1 for t in traj if t["engagement"] >= 0.55) / n
        low_stress_score = sum(1 for t in traj if t["stress"] <= 0.50) / n
        correct_rate = sum(1 for t in traj if t.get("correct") is True) / n

        # FIX: calibrated for 4 steps — reward 2+ unique content_types (not unique (type,diff) pairs)
        unique_types = len(set(t["action"]["content_type"] for t in traj))
        variety_score = min(1.0, (unique_types - 1) / 2.0)  # 1 type=0.0, 2=0.5, 3+=1.0

        score = (
            0.35 * engagement_score
            + 0.35 * low_stress_score
            + 0.20 * correct_rate
            + 0.10 * variety_score
        )
        score = round(max(0.0, min(1.0, score)), 4)
        return GraderResult(
            task_id="easy",
            score=score,
            breakdown={
                "engagement_score": round(engagement_score, 3),
                "low_stress_score": round(low_stress_score, 3),
                "correct_rate":     round(correct_rate, 3),
                "variety_score":    round(variety_score, 3),
            },
            passed=score >= 0.50
        )

    # ── Task 2: Distress Response (Medium, 7 steps) ──────────────────── #
    def _grade_medium(self, state: State) -> GraderResult:
        """
        Objective: Keep stress under control across 7 steps.
        Prevention is rewarded MORE than cure.

        Design principle (PhD fix):
          A perfect agent that keeps stress below 0.38 ALL 7 steps scores HIGHER
          than a mediocre agent that lets stress spike to 0.75 then recovers.
          The old grader had this backwards — recovery_score dominated, so
          agents that caused spikes then fixed them outscored agents that
          prevented spikes entirely.

        Components (trajectory-only, no hidden state):
          - prevention_score: avg_stress kept near 0.35 (full credit) vs 0.55 (zero)
          - control_score:    % of steps where stress <= 0.50
          - recovery_bonus:   if stress DID spike > 0.55, reward bringing it back
                              if stress never spiked, full bonus (prevention rewarded)
          - speed_bonus:      if spike occurred, how fast was recovery
          - learning_score:   avg learning per step (prevents break-spam)
        """
        traj = state.trajectory
        if not traj:
            return GraderResult(task_id="medium", score=0.0, breakdown={}, passed=False)

        stresses  = [t["stress"]   for t in traj]
        learnings = [t["learning"] for t in traj]
        n = len(stresses)

        avg_stress   = sum(stresses) / n
        peak_stress  = max(stresses)
        final_stress = stresses[-1]

        # ── Prevention: was stress kept low throughout? ───────────────── #
        # Full credit: avg_stress <= 0.35. Zero credit: avg_stress >= 0.55.
        prevention_score = max(0.0, min(1.0, 1.0 - (avg_stress - 0.35) / 0.20))

        # ── Control: % of steps under 0.50 ───────────────────────────── #
        control_score = sum(1 for s in stresses if s <= 0.50) / n

        # ── Recovery bonus: only relevant if stress spiked above 0.55 ─── #
        spike_steps = [i for i, s in enumerate(stresses) if s > 0.55]
        if spike_steps:
            first_spike = spike_steps[0]
            # How much did stress drop from spike to end?
            recovery = max(0.0, peak_stress - final_stress)
            recovery_bonus = min(1.0, recovery / 0.30)
            # How quickly did it recover?
            threshold = peak_stress - 0.10
            steps_to_recover = next(
                (i + 1 for i, s in enumerate(stresses[first_spike:]) if s < threshold),
                n
            )
            speed_bonus = max(0.0, 1.0 - (steps_to_recover - 1) / 3.0)
        else:
            # No spike → prevention was perfect → full bonus for both
            recovery_bonus = 1.0
            speed_bonus    = 1.0

        # ── Learning: avg per step (penalises break-spam) ─────────────── #
        avg_learning  = sum(learnings) / n
        learning_score = min(1.0, avg_learning / 0.07)

        score = (
            0.35 * prevention_score
            + 0.25 * control_score
            + 0.20 * recovery_bonus
            + 0.10 * speed_bonus
            + 0.10 * learning_score
        )

        # Learning gate: a therapeutic session with near-zero learning is not
        # a good session regardless of stress levels. Clinically grounded —
        # calm-but-passive is not the goal. Penalises break-spam strategies.
        avg_learning = sum(learnings) / n
        if avg_learning < 0.04:
            score *= 0.55

        score = round(max(0.0, min(1.0, score)), 4)
        return GraderResult(
            task_id="medium",
            score=score,
            breakdown={
                "prevention_score":  round(prevention_score, 3),
                "control_score":     round(control_score, 3),
                "recovery_bonus":    round(recovery_bonus, 3),
                "speed_bonus":       round(speed_bonus, 3),
                "learning_score":    round(learning_score, 3),
                "avg_stress":        round(avg_stress, 3),
                "peak_stress":       round(peak_stress, 3),
                "had_spike":         bool(spike_steps),
                "avg_learning":      round(avg_learning, 4),
            },
            passed=score >= 0.45
        )

    # ── Task 3: Full Session Optimization (Hard, 12 steps) ───────────── #
    def _grade_hard(self, state: State) -> GraderResult:
        """
        Objective: maximize learning over 12 steps while managing stress,
        improving adaptation, and deliberately choosing varied subjects.

        FIX: variety_score now measures action.subject choices (agent-controlled),
             not random question subjects from the environment.

        A constant-action policy CANNOT score well:
          - variety_score=0 if agent never sets action.subject
          - adaptation_score=0.5 if stress doesn't improve in second half
          - consistency_score penalizes volatile stress
        """
        traj = state.trajectory
        if not traj:
            return GraderResult(task_id="hard", score=0.0, breakdown={}, passed=False)

        n = len(traj)
        stresses    = [t["stress"]     for t in traj]
        learnings   = [t["learning"]   for t in traj]
        engagements = [t["engagement"] for t in traj]

        # Learning: normalized by theoretical max per step
        max_possible = 0.15 * n
        total_learning = sum(learnings)
        learning_score = min(1.0, total_learning / max_possible)

        # Stress: strict — avg > 0.40 is heavily penalized
        avg_stress = sum(stresses) / n
        stress_score = max(0.0, min(1.0, 1.0 - avg_stress / 0.40))

        # Engagement
        avg_engagement = sum(engagements) / n

        # Adaptation: second half stress must improve on first half
        mid = n // 2
        first_half_stress  = sum(stresses[:mid])  / max(1, mid)
        second_half_stress = sum(stresses[mid:])  / max(1, n - mid)
        improvement = first_half_stress - second_half_stress
        adaptation_score = min(1.0, max(0.0, 0.5 + improvement * 2.5))

        # Consistency: low volatility
        if n > 1:
            volatility = sum(abs(stresses[i] - stresses[i-1]) for i in range(1, n)) / (n - 1)
        else:
            volatility = 0.0
        consistency_score = max(0.0, 1.0 - volatility * 3.5)

        # FIX: Variety = agent-chosen subjects from action.subject field
        # Agent must deliberately diversify — this is now a skill, not luck
        chosen_subjects = set(
            t["action"].get("subject")
            for t in traj
            if t["action"].get("subject") is not None
        )
        variety_score = min(1.0, len(chosen_subjects) / 3.0)  # 3+ = full score

        # Overload events: stress spike > 0.08 in one step (trajectory-only)
        overload_events = sum(
            1 for i in range(1, n) if stresses[i] - stresses[i-1] > 0.08
        )
        overload_penalty = min(0.30, overload_events * 0.06)

        score = (
            0.28 * learning_score
            + 0.22 * stress_score
            + 0.18 * avg_engagement
            + 0.15 * adaptation_score
            + 0.10 * consistency_score
            + 0.07 * variety_score
            - overload_penalty
        )
        score = round(max(0.0, min(1.0, score)), 4)
        return GraderResult(
            task_id="hard",
            score=score,
            breakdown={
                "learning_score":     round(learning_score, 3),
                "stress_score":       round(stress_score, 3),
                "engagement_score":   round(avg_engagement, 3),
                "adaptation_score":   round(adaptation_score, 3),
                "consistency_score":  round(consistency_score, 3),
                "variety_score":      round(variety_score, 3),
                "overload_penalty":   round(overload_penalty, 3),
                "avg_stress":         round(avg_stress, 3),
                "total_learning":     round(total_learning, 4),
                "chosen_subjects":    len(chosen_subjects),
            },
            passed=score >= 0.40
        )
