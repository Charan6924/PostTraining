"""Binary final-answer reward shared by PPO and GRPO."""


def correctness_reward(response, expected_answer):
    """Return 1.0 for correctness, otherwise 0.0, including invalid answers."""
    raise NotImplementedError("Implement numeric normalization and answer checking.")
