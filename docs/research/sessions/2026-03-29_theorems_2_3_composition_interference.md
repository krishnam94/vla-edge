# Theorems 2 and 3: Composition Safety and Contract Interference

**Date**: 2026-03-29
**Agent**: Formal Methods Researcher
**Purpose**: Rigorous theorem statements and proof sketches for the SafeContract paper (C2 contribution)

---

## Notation and Setup

We work in discrete time (VLA inference at frequency f Hz). This is a deliberate departure from the continuous-time CBF literature (Ames et al. 2019, Glotfelter et al. 2018) - VLA policies are called once per timestep and return a complete action vector.

**Definition 1 (Action Space).** Let A = R^d be the d-dimensional action space of a VLA policy, where each dimension corresponds to a joint command or end-effector coordinate.

**Definition 2 (Safety Contract).** A safety contract C = (S, phi) consists of:
- A *safe set* S subset of A, a closed convex polytope
- A *projection operator* phi: A -> S that maps any action to the nearest point in S

For polytopic safe sets (hyperrectangles in our case), phi is componentwise clipping:

  phi(a)_i = clip(a_i, lo_i, hi_i)

**Definition 3 (State-Dependent Contract).** A *velocity contract* is state-dependent: given the previous action a_{t-1}, the safe set at time t is

  S_vel(a_{t-1}) = { a in A : ||a - a_{t-1}||_inf <= v_max }

This is a hyperrectangle centered at a_{t-1} with half-width v_max.

**Definition 4 (Contract Composition).** Given contracts C_1 = (S_1, phi_1) and C_2 = (S_2, phi_2), their composition C_1 compose C_2 applies phi_2 after phi_1:

  (phi_1 compose phi_2)(a) = phi_2(phi_1(a))

The *feasible set* of the composition is F = S_1 intersect S_2.

---

## Theorem 2: Composition Safety

**Theorem 2 (Composition Safety).** Let C_1 = (S_1, phi_1) and C_2 = (S_2, phi_2) be safety contracts where S_1 and S_2 are closed convex sets, and phi_1, phi_2 are the Euclidean projections onto S_1, S_2 respectively. If the feasible set F = S_1 intersect S_2 is non-empty, then:

(i) **Safety**: For any action a in A, the composed output phi_2(phi_1(a)) in S_2. That is, the output always satisfies the *last* applied contract.

(ii) **Mutual Safety**: If additionally phi_2(phi_1(a)) in S_1 for all a (i.e., projecting onto S_2 does not leave S_1), then phi_2(phi_1(a)) in F for all a.

(iii) **Convergence under alternating projections**: When S_1 and S_2 are closed convex sets with non-empty intersection, the alternating projection sequence phi_1, phi_2, phi_1, phi_2, ... converges to a point in F. In our case, a single composition phi_2(phi_1(a)) already lands in S_2, and for hyperrectangular contracts, a single pass suffices to land in F.

### Proof Sketch

**(i)** is immediate: phi_2 maps into S_2 by definition. This holds regardless of whether S_1 intersect S_2 is empty.

**(ii)** The key insight is that for *axis-aligned hyperrectangles* (our case - action bounds and velocity bounds are per-dimension), the composition is order-independent and lands in the intersection in one pass.

**Claim.** When S_1 = prod_i [lo1_i, hi1_i] and S_2 = prod_i [lo2_i, hi2_i] are hyperrectangles, componentwise clipping satisfies:

  phi_2(phi_1(a))_i = clip(clip(a_i, lo1_i, hi1_i), lo2_i, hi2_i)
                     = clip(a_i, max(lo1_i, lo2_i), min(hi1_i, hi2_i))

*Proof.* Consider dimension i. Let b = clip(a_i, lo1_i, hi1_i), so b in [lo1_i, hi1_i]. Then clip(b, lo2_i, hi2_i). Since F_i = [max(lo1_i, lo2_i), min(hi1_i, hi2_i)] is non-empty by assumption, we verify by cases:

- If a_i < max(lo1_i, lo2_i): The first clip gives max(lo1_i, a_i). If lo1_i >= lo2_i, we get lo1_i, and the second clip is trivially lo1_i in F_i. If lo1_i < lo2_i, the first clip may give some b in [lo1_i, lo2_i), and the second clip maps it to lo2_i = max(lo1_i, lo2_i).
- If a_i > min(hi1_i, hi2_i): Symmetric argument.
- If a_i in F_i: Both clips are identity. QED.

This is the nested-clip-equals-intersection-clip identity. It means sequential application of hyperrectangular contracts is equivalent to a single clip to their intersection, making composition both safe and order-independent.

**(iii)** For general convex sets (not axis-aligned), this follows from the von Neumann/Halperin alternating projections theorem: if S_1, S_2 are closed convex subsets of a Hilbert space with S_1 intersect S_2 != empty, the sequence of alternating projections converges in norm to the projection onto S_1 intersect S_2. In practice, VLA contracts are hyperrectangular, so (ii) gives us one-shot convergence.

**Reference:** The alternating projections result dates to von Neumann (1950) and was generalized by Halperin (1962). For our discrete, polytopic setting, Bauschke & Borwein (1996) "On Projection Algorithms for Solving Convex Feasibility Problems" provides the definitive treatment. The connection to barrier function composition in continuous time is developed in Glotfelter et al. (2018), who use min/max operations on nonsmooth CBFs - our hyperrectangle clipping is the discrete-time analog of their min-composition.

### Practical Implication

For the @safety_contract decorator, stacking two decorators:

```python
@safety_contract(bounds=[-1, 1])        # C1
@safety_contract(velocity_max=0.1)      # C2
def predict(obs) -> np.ndarray:
    return model(obs)
```

is equivalent to a single decorator with the intersection constraints, provided the intersection is non-empty. The application order does not matter for hyperrectangular contracts.

---

## Theorem 3: Interference and Deadlock

The interesting case is when contracts can conflict. This arises because velocity contracts are *state-dependent* - the feasible set shifts with each timestep.

**Theorem 3 (Interference Conditions).** Let C_bounds = (S_bounds, phi_bounds) be a workspace/action-range contract with fixed safe set S_bounds = prod_i [lo_i, hi_i], and let C_vel = (S_vel(a_{t-1}), phi_vel) be a velocity contract with state-dependent safe set S_vel(a_{t-1}) = prod_i [a_{t-1,i} - v_max, a_{t-1,i} + v_max].

Define the *feasibility margin* at state a_{t-1} as:

  mu(a_{t-1}) = min_i ( min(hi_i, a_{t-1,i} + v_max) - max(lo_i, a_{t-1,i} - v_max) )

Then:

**(i) Non-empty feasible set:** F(a_{t-1}) = S_bounds intersect S_vel(a_{t-1}) is non-empty if and only if mu(a_{t-1}) >= 0.

**(ii) Deadlock (empty feasible set):** F(a_{t-1}) = empty if and only if there exists a dimension i such that:

  a_{t-1,i} + v_max < lo_i   OR   a_{t-1,i} - v_max > hi_i

That is, the robot's previous action is so far outside the workspace bounds that even maximum velocity cannot bring it back within one timestep.

**(iii) Deadlock is unreachable from safe states:** If a_{t-1} in S_bounds and v_max > 0, then mu(a_{t-1}) >= min(v_max, min_i(hi_i - lo_i)) > 0. In particular, deadlock CANNOT occur if the system starts in the workspace bounds and the velocity limit is positive.

**(iv) Shrinking margin near boundaries:** For a_{t-1,i} in [lo_i, hi_i], the feasibility margin in dimension i is:

  mu_i(a_{t-1,i}) = min(hi_i, a_{t-1,i} + v_max) - max(lo_i, a_{t-1,i} - v_max)
                   = min(hi_i - lo_i, 2*v_max, a_{t-1,i} - lo_i + v_max, hi_i - a_{t-1,i} + v_max)

This is minimized when a_{t-1} is at the boundary of S_bounds, where:

  mu_i(lo_i) = mu_i(hi_i) = min(hi_i - lo_i, v_max)

The feasible set never vanishes but may become a single point (when hi_i - lo_i < v_max and a_{t-1,i} is at the boundary), restricting the policy to zero effective velocity in that dimension.

### Proof of (ii)

F(a_{t-1}) = empty iff there exists i such that [lo_i, hi_i] intersect [a_{t-1,i} - v_max, a_{t-1,i} + v_max] = empty (since both are hyperrectangles, their intersection is empty iff it is empty in at least one dimension). Two intervals [lo_i, hi_i] and [a_{t-1,i} - v_max, a_{t-1,i} + v_max] are disjoint iff a_{t-1,i} + v_max < lo_i or a_{t-1,i} - v_max > hi_i. QED.

### Proof of (iii)

If a_{t-1,i} in [lo_i, hi_i], then a_{t-1,i} >= lo_i, so a_{t-1,i} + v_max >= lo_i + v_max > lo_i. Also a_{t-1,i} <= hi_i, so a_{t-1,i} - v_max <= hi_i - v_max < hi_i. Therefore [a_{t-1,i} - v_max, a_{t-1,i} + v_max] intersect [lo_i, hi_i] contains at least a_{t-1,i} itself. QED.

### Proof of (iv)

Direct computation. When a_{t-1,i} = lo_i:
  mu_i = min(hi_i, lo_i + v_max) - max(lo_i, lo_i - v_max) = min(hi_i, lo_i + v_max) - lo_i = min(hi_i - lo_i, v_max)

Similarly for a_{t-1,i} = hi_i. The margin is maximized in the interior of [lo_i, hi_i], where it equals min(hi_i - lo_i, 2*v_max). QED.

### The Concrete Example from the Problem Statement

Workspace bounds: [0, 0.5]. Velocity limit: v_max = 0.1. Previous action: a_{t-1} = 0.49.

- S_bounds = [0, 0.5]
- S_vel(0.49) = [0.39, 0.59]
- F = [0.39, 0.5] (width 0.11 - non-empty, safe)
- mu = 0.11

The robot can move anywhere in [0.39, 0.5]. The policy is constrained but not deadlocked.

When does it become empty? Only if a_{t-1} - v_max > 0.5, i.e., a_{t-1} > 0.6. But if the contracts have been enforced since the start, a_{t-1} <= 0.5, so a_{t-1} - v_max <= 0.4 < 0.5. Deadlock is impossible.

**The only way to reach deadlock is to start outside the workspace bounds** (e.g., the robot was manually moved, or a previous controller did not enforce bounds). This motivates an initialization check in the @safety_contract decorator.

---

## Corollary: Interference-Free Composition Condition

**Corollary 1 (Interference-Free Composition).** A set of hyperrectangular contracts {C_k}_{k=1}^K with safe sets S_k = prod_i [lo_k,i, hi_k,i] is *interference-free* if and only if for each dimension i:

  max_k lo_k,i <= min_k hi_k,i

For state-dependent contracts (velocity, acceleration), this must hold for all reachable states. Specifically, composing a fixed workspace contract S_bounds with a velocity contract S_vel is interference-free for all states reachable under the composed system, as shown in Theorem 3(iii).

---

## Connection to Existing Theory

### Relationship to Glotfelter et al. (2018) Boolean CBF Composition

Glotfelter's work composes continuous-time CBFs using min/max:
- h_AND(x) = min(h_1(x), h_2(x)) for conjunction (both constraints)
- h_OR(x) = max(h_1(x), h_2(x)) for disjunction (either constraint)

Our setting is the discrete-time specialization where:
- The safe sets are hyperrectangles (axis-aligned CBFs in each dimension)
- The "controller synthesis via QP" becomes simple componentwise clipping
- The nonsmooth generalized gradient machinery is unnecessary because clip is already well-defined

The key insight Glotfelter identifies - that the AND-composition h_AND >= 0 iff h_1 >= 0 AND h_2 >= 0, which corresponds to the intersection of safe sets - is exactly our Theorem 2. The machinery differs (QP vs clipping) but the core structure is identical.

### Relationship to Mallozzi et al. (2023) Contract-Based Robotic Missions

Mallozzi et al. define assume-guarantee contracts where:
- *Assumptions* A: what the environment provides (e.g., robot starts in workspace)
- *Guarantees* G: what the component ensures (e.g., actions stay in bounds)

Their composition rule: C_1 compose C_2 is compatible if A_1 intersect A_2 != empty (the assumptions are jointly satisfiable). Our Theorem 3(ii) is exactly the compatibility check: deadlock = incompatible contracts = empty intersection of assumptions.

Their key contribution - contract refinement checking via LTL - operates at the mission level (reach goal, avoid obstacles over time). Ours operates at the action level (each timestep's output satisfies bounds). The two are complementary.

### Relationship to Ames et al. (2019) CBF-QP Feasibility

Ames identifies that CBF-QP infeasibility arises when the safe set and the input constraint set have empty intersection. Their solution: optimal-decay CBFs that relax the barrier constraint. Our analog: when the feasibility margin mu is small, the policy has reduced freedom but the system remains safe. We never need relaxation because Theorem 3(iii) guarantees non-empty feasible sets for all reachable states.

### Connection to Polytope Geometry (Minkowski Operations)

The velocity contract's feasible set can be written as a Minkowski sum:

  S_vel(a_{t-1}) = {a_{t-1}} oplus B_inf(v_max)

where B_inf(v_max) is the L-infinity ball of radius v_max. The feasible set under composition is:

  F(a_{t-1}) = S_bounds intersect ({a_{t-1}} oplus B_inf(v_max))

This is the intersection of a fixed polytope with a translated polytope - a standard problem in computational geometry. For hyperrectangles, this intersection is itself a hyperrectangle (or empty), computable in O(d) time. For general polytopes, computing the intersection is NP-hard in the number of facets (Tiwary, 2008), but our axis-aligned structure avoids this complexity entirely.

### Relationship to VLSA/AEGIS (2024) CBF Safety Layer for VLAs

AEGIS (arXiv:2512.11891) formulates a CBF-based safety constraint layer for VLA models. Their approach solves a QP at each timestep to find the safe action closest to the VLA output. Our approach is strictly simpler: componentwise clipping. The tradeoff:

| | AEGIS (CBF-QP) | SafeContract (Clipping) |
|---|---|---|
| Constraint expressiveness | Arbitrary convex | Hyperrectangular |
| Composition theory | Inherits CBF composition | Theorem 2 (exact, one-shot) |
| Runtime | O(d^3) QP solve | O(d) clip |
| Deadlock handling | QP infeasibility detection | Theorem 3 (provably unreachable) |
| Deployment overhead | Requires QP solver | Pure numpy |

For VLA deployment on edge hardware (Jetson, Mac), the O(d) clipping with provable safety is preferable to the O(d^3) QP with more expressive but harder-to-verify constraints.

---

## Summary for Paper

**Theorem 2** establishes that stacking hyperrectangular contracts is equivalent to intersecting them, with one-shot convergence (no iterative projection needed). This justifies the decorator-stacking pattern.

**Theorem 3** characterizes exactly when contract interference causes deadlock (empty feasible set) and proves it is unreachable from safe initial states under positive velocity limits. This is the paper's strongest theoretical contribution - it means users can freely compose workspace, velocity, and action-range contracts without worrying about deadlock, as long as the robot starts in-bounds.

**Corollary 1** gives a simple check (per-dimension interval overlap) for verifying interference-freedom at contract definition time, before any inference is run.

---

## Sources

- [Ames et al. 2019 - Control Barrier Functions: Theory and Applications](https://arxiv.org/abs/1903.11199) - Foundation of CBF theory, CBF-QP formulation, feasibility analysis
- [Glotfelter et al. 2018 - Boolean Composability via Nonsmooth CBFs](https://ieeexplore.ieee.org/document/8511471/) - min/max composition of barrier functions for multi-robot safety
- [Mallozzi et al. 2023 - Correct-by-Construction Robotic Missions Using Contracts](https://arxiv.org/abs/2306.08144) - Assume-guarantee contracts for robotic mission design
- [VLSA/AEGIS 2024 - VLA with Plug-and-Play Safety Constraint Layer](https://arxiv.org/abs/2512.11891) - CBF-QP safety layer for VLA models
- [SafeVLA 2025 - Safety Alignment via Constrained Learning](https://arxiv.org/abs/2503.03480) - Training-time safety for VLAs (complementary approach)
- [Zeng et al. 2021 - Safety-Critical Control using Optimal-decay CBFs](https://hybrid-robotics.berkeley.edu/publications/ACC2021_CBF-Pointwise-Feasibility.pdf) - Feasibility guarantees for CBF-QPs
- [Bauschke & Borwein 1996 - On Projection Algorithms for Solving Convex Feasibility Problems](https://link.springer.com/article/10.1007/BF02142692) - Alternating projections convergence theory
- [Nuzzo et al. 2017 - Stochastic Assume-Guarantee Contracts for CPS](https://arxiv.org/abs/1705.09316) - Contract compatibility and consistency checking
- [Tiwary 2008 - On the Hardness of Minkowski Sum and Polytope Intersection](https://kam.mff.cuni.cz/~hansraj/publications/minkowski_dcg.pdf) - Complexity of polytope intersection
- [Grover et al. 2023 - The Before, During and After of Multi-Robot Deadlock](https://jaskaransgrover.github.io/files/beforedurinnafterdeadlock.pdf) - Deadlock characterization with CBFs
