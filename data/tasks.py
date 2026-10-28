"""Dataset task names and degradation descriptions."""

TASKS = ("Low_light", "Over_exposure", "IR_Low_contrast", "IR_Noise")
TASK_KEYS = {
    "Low_light": "low_light",
    "Over_exposure": "over_exposure",
    "IR_Low_contrast": "ir_low_contrast",
    "IR_Noise": "ir_noise",
}
DEFAULT_PROMPTS = {
    "low_light": "Fuse the infrared and visible images; the visible image has low-light degradation.",
    "over_exposure": "Fuse the infrared and visible images; the visible image is overexposed.",
    "ir_low_contrast": "Fuse the infrared and visible images; the infrared image has low contrast.",
    "ir_noise": "Fuse the infrared and visible images; the infrared image is affected by noise.",
}
GENERIC_PROMPT = "Fuse the infrared and visible images while preserving salient targets and structural details."
