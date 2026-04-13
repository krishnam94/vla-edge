# Sprint: Apr 14 - May 29 (CoRL deadline)

Full plan: `~/Desktop/wiki/master-execution-plan-apr-may.md`

## Week 1: Apr 14-20 - Foundation (~16-20 hrs)
- [ ] 1.1: Create examples/ directory (3 runnable scripts, no hardware needed)
- [ ] 1.2: Simplified top-level API (SafetyGuard, safety_contract at package root)
- [ ] 1.3: YAML config for SafetyGuard (from_yaml + configs/pusht.yaml)
- [ ] 1.4: **MVE validation** - run complementarity test on existing PushT JSON data (GO/NO-GO)
- [ ] 1.5: README rewrite (practitioner-first, link to examples)
- [ ] 1.6: Fix CI (ruff lint)

## Week 2: Apr 21-24 - Ship Before Japan (~12-16 hrs)
- [ ] 2.1: RFC issue on huggingface/lerobot (Apr 21)
- [ ] 2.2: Implement SafetyClipProcessorStep (~150 lines, Apr 22)
- [ ] 2.3: Submit PR to huggingface/lerobot (Apr 23)
- [ ] 2.4: Outline CoRL Section 3 from wiki articles (Apr 23-24)
- [ ] 2.5: Push all vla-edge improvements to GitHub

## Weeks 3-4: Apr 24 - May 6 - Japan (writing only, ~12-16 hrs)
- [ ] 3.1: CoRL Section 3 full draft (architecture, composition rules)
- [ ] 3.2: Related work paper notes (SAFE, FIPER, CoVer, Sentinel, SafeVLA)
- [ ] 3.3: Respond to LeRobot PR reviews (async, 30 min/day)
- [ ] 3.4: Draft CoRL abstract

## Weeks 5-6: May 7-18 - CoRL Experiments (~60-80 hrs)
- [ ] Infrastructure: Clone CoVer-VLA + FIPER, verify LIBERO (May 7-8)
- [ ] GATE 1 (May 8): Which external tools work? Decide experiment scope.
- [ ] Exp 1: Failure coverage matrix - 6 failure types x 6 monitoring configs ($50-80)
- [ ] Exp 2: CUSUM shift detection - 150 rollouts, compare vs FIPER
- [ ] Exp 3: Latency budget validation - profile all layers
- [ ] GATE 2 (May 14): Are results publishable? Decide paper scope.
- [ ] Exp 4: Safety-performance Pareto (OPTIONAL, only if ahead of schedule)
- [ ] Buffer days May 17-18

## Week 7: May 19-29 - CoRL Writing + Submission (~50-60 hrs)
- [ ] May 19-20: Draft Sections 4 (experiments) + 5 (limitations)
- [ ] May 21-22: Draft Sections 1 (intro) + 2 (related work) + 6 (conclusion)
- [ ] May 23-24: Polish Section 3 + create all figures (5 figures)
- [ ] May 25: Internal review + abstract submission (deadline May 26)
- [ ] May 26-28: Revisions, figure polish, reference check, proofread
- [ ] May 29: Submit on OpenReview

## ICRA WS (if accepted May 1)
- [ ] Update with ALOHA AUROC figure (3-architecture bar chart)
- [ ] Book VFS Schengen visa appointment (May 7-12 window)
- [ ] Prepare poster

## Backlog (post-CoRL, May 30+)
- [ ] Phase 2: lerobot-safety-guard standalone package (1-2 weeks)
- [ ] Manning Ch10 first draft (architecture section feeds directly)
- [ ] Real robot validation (SO-101 or Siddharth's lab)
- [ ] Workshop papers: RSS SAFE-ROL (~Aug), NeurIPS workshop (~Sep)
- [ ] REWS composite score implementation
- [ ] EU AI Act compliance logging
