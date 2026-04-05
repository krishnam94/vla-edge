# Integration Checklist for VLA Model Adapters

Check EVERY item when integrating a new VLA model or connecting to a new environment.
This checklist exists because we missed action unnormalization in the LIBERO closed-loop
experiment (Lesson 010), causing 0% task success from incorrect action scaling.

## Action Space

- [ ] **Normalization mode**: What normalization does the model use? (MEAN_STD, MIN_MAX, QUANTILES, IDENTITY)
- [ ] **Dataset stats**: Where are mean/std/min/max stored? (model config, separate file, dataset)
- [ ] **Unnormalization**: Is the output unnormalized before sending to environment/robot?
- [ ] **Action range**: What range does the environment expect? ([-1,1], radians, meters?)
- [ ] **Action dimensions**: Do model output dims match env action dims?
- [ ] **Delta vs absolute**: Does the model output delta actions or absolute positions?
- [ ] **Gripper handling**: Is gripper continuous or binary? What dim index?

## Observation Space

- [ ] **Image format**: (H,W,C) uint8 or (C,H,W) float [0,1]? RGB or BGR?
- [ ] **Image size**: Does model expect same resolution as env provides?
- [ ] **Image keys**: What observation keys does the model expect? (camera names vary)
- [ ] **State normalization**: Is robot state normalized before feeding to model?
- [ ] **Language tokenization**: Raw string or pre-tokenized? Attention mask dtype?

## Control Loop

- [ ] **Action chunking**: Does the model cache actions? Must call policy.reset() between episodes?
- [ ] **Control frequency**: Steps per second expected by env vs model inference time
- [ ] **Queue flushing**: Lesson 007 - SmolVLA caches 50 actions, must flush before new obs

## Safety

- [ ] **Violations tracked on raw model output** (before unnormalization/clipping)
- [ ] **Safety contract applied on unnormalized actions** (in env's action space)
- [ ] **Velocity limits appropriate** for the control frequency and action space

## SmolVLA + LIBERO Specific Bugs Found (Lesson 014)

These 3 critical bugs caused 0% success and were found via root cause analysis:

1. **State representation**: LIBERO gives joint_pos(7) + gripper(2). SmolVLA expects
   eef_pos(3) + axis_angle(3) + gripper(2). Must convert using robot0_eef_pos,
   robot0_eef_quat -> quaternion-to-axis-angle conversion.

2. **State normalization**: SmolVLA uses MEAN_STD normalization on state inputs.
   Must normalize before feeding to model (use lerobot's NormalizerProcessorStep
   or extract stats from checkpoint).

3. **Image flip**: LIBERO images must be flipped 180 degrees (both H and W)
   before feeding to SmolVLA. The training pipeline does this via
   LiberoProcessorStep: `torch.flip(img, dims=[2, 3])`.

Best practice: Use lerobot's own preprocessing pipeline instead of reimplementing.
