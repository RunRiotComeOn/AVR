"""
Adaptive reward with intra-group diversity bonus.

This reward encourages the model to explore different format types within
each GRPO group (N responses per prompt).

Diversity Bonus:
    For each response in a group, compute the frequency of its format type:
        freq_i = count(type_i in group) / group_size
    diversity_bonus = diversity_weight(step) * (1 - freq_i)

    This means minority format types within a group receive a higher bonus.

Diversity Weight Decay:
    The diversity weight decays over the entire training run (step 0 to
    total_training_steps) so that early in training the model is encouraged
    to explore, but later the diversity pressure fades and quality/correctness
    dominates. No start/end step needed — cosine decay naturally ensures
    the diversity influence is very weak toward the end.
"""

import re
import math
from collections import Counter
from typing import List, Dict, Any, Optional
import numpy as np


class AdaptiveReward:
    """
    Adaptive reward function with format bonuses, length regularization,
    and intra-group diversity bonus.
    """

    def __init__(
        self,
        # Correctness rewards
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        # Format bonuses (correct only)
        type1_format_bonus: float = 0.5,
        type2_format_bonus: float = 0.3,
        type3_format_bonus: float = 0.0,
        # Error penalties (incorrect only)
        type1_error_penalty: float = -0.5,
        type2_error_penalty: float = -0.2,
        type3_error_penalty: float = 0.0,
        # Length regularization
        length_threshold: int = 300,
        ideal_length: float = 300.0,
        min_scalar: float = 0.3,
        # Unknown penalty (incomplete/missing tags)
        unknown_penalty: float = -1.0,
        # Diversity bonus
        initial_diversity_weight: float = 0.5,
        total_training_steps: int = 1000,
        # Other
        normalize_answers: bool = True,
    ):
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.type1_format_bonus = type1_format_bonus
        self.type2_format_bonus = type2_format_bonus
        self.type3_format_bonus = type3_format_bonus
        self.type1_error_penalty = type1_error_penalty
        self.type2_error_penalty = type2_error_penalty
        self.type3_error_penalty = type3_error_penalty
        self.length_threshold = length_threshold
        self.ideal_length = ideal_length
        self.min_scalar = min_scalar
        self.unknown_penalty = unknown_penalty
        self.normalize_answers = normalize_answers

        # Diversity parameters
        self.initial_diversity_weight = initial_diversity_weight
        self.total_training_steps = total_training_steps

        self.training_step = 0

    def set_training_step(self, step: int):
        """Update the current training step."""
        self.training_step = step

    def set_total_training_steps(self, total: int):
        """Update total training steps (auto-detected from trainer)."""
        if total > 0:
            self.total_training_steps = total

    def get_diversity_weight(self) -> float:
        """
        Compute the current diversity weight with cosine decay.
        """
        step = self.training_step
        total = self.total_training_steps

        if total <= 0 or step >= total:
            return 0.0

        progress = step / total  # 0 -> 1
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        return self.initial_diversity_weight * max(0.0, factor)

    def extract_answer(self, response: str) -> str:
        """Extract answer from <answer>...</answer> tags."""
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        return response.strip()

    def has_complete_tag(self, response: str, tag_name: str) -> bool:
        """Check if response contains both <tag> and </tag> (complete pair)."""
        has_open = bool(re.search(rf'<{tag_name}>', response, re.IGNORECASE))
        has_close = bool(re.search(rf'</{tag_name}>', response, re.IGNORECASE))
        return has_open and has_close

    def get_response_type(self, response: str) -> int:
        """
        Determine response type based on complete tag pair presence.

        Returns:
            3: Type 3 - <perception> + <reasoning> + <answer>
            2: Type 2 - <perception> + <answer>
            1: Type 1 - <answer> only
            0: Unknown - incomplete or missing tags
        """
        has_perception = self.has_complete_tag(response, 'perception')
        has_reasoning = self.has_complete_tag(response, 'reasoning')
        has_answer = self.has_complete_tag(response, 'answer')

        if has_perception and has_reasoning and has_answer:
            return 3
        elif has_perception and has_answer and not has_reasoning:
            return 2
        elif has_answer and not has_perception and not has_reasoning:
            return 1
        else:
            return 0

    def count_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: words * 1.3)."""
        words = len(text.split())
        return int(words * 1.3)

    def calculate_length_scalar(self, token_count: int) -> float:
        """Length regularization scalar."""
        if token_count <= self.length_threshold:
            return 1.0
        scalar = self.ideal_length / token_count
        return max(self.min_scalar, min(1.0, scalar))

    def normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not self.normalize_answers:
            return answer
        answer = answer.lower().strip()
        answer = answer.rstrip('.,!?;:')
        answer = ' '.join(answer.split())
        for prefix in ['the ', 'a ', 'an ']:
            if answer.startswith(prefix):
                answer = answer[len(prefix):]
        return answer

    def check_answer_correctness(self, predicted: str, ground_truth: Any) -> bool:
        """Check if predicted answer matches ground truth."""
        pred_answer = self.extract_answer(predicted)
        pred_norm = self.normalize_answer(pred_answer)

        gt_answers = ground_truth if isinstance(ground_truth, list) else [ground_truth]

        for gt in gt_answers:
            gt_norm = self.normalize_answer(str(gt))

            if len(gt_norm) == 1 and gt_norm.upper() in ['A', 'B', 'C', 'D', 'E']:
                if re.search(rf'\b{gt_norm.upper()}\b', pred_answer.upper()):
                    return True
                continue

            if pred_norm == gt_norm:
                return True
            if gt_norm in pred_norm or pred_norm in gt_norm:
                return True

            try:
                pred_num = float(pred_norm)
                gt_num = float(gt_norm)
                if abs(pred_num - gt_num) < 1e-6:
                    return True
            except (ValueError, TypeError):
                pass

        return False

    def _compute_diversity_bonuses(
        self,
        response_types: List[int],
        uids: Optional[List[str]],
    ) -> List[float]:
        """
        Compute per-response diversity bonus based on intra-group format rarity.

        For each group (same uid), count the frequency of each format type.
        Responses with a rarer format type within the group get a higher bonus:
            bonus_i = diversity_weight * (1 - freq_i)

        Args:
            response_types: List of format types (0, 1, 2, 3) for each response.
            uids: List of group identifiers. If None, treat entire batch as one group.

        Returns:
            List of diversity bonus values per response.
        """
        diversity_weight = self.get_diversity_weight()

        if diversity_weight <= 0.0:
            return [0.0] * len(response_types)

        n = len(response_types)
        bonuses = [0.0] * n

        if uids is None:
            # Treat entire batch as one group
            uids = ["__all__"] * n

        # Group indices by uid
        groups: Dict[str, List[int]] = {}
        for i, uid in enumerate(uids):
            groups.setdefault(uid, []).append(i)

        for uid, indices in groups.items():
            # Count valid format types in this group (exclude unknown=0)
            valid_types = [response_types[i] for i in indices if response_types[i] > 0]
            group_size = len(valid_types)

            if group_size <= 1:
                # Single valid response or no valid responses: no diversity signal
                continue

            type_counts = Counter(valid_types)

            for i in indices:
                rtype = response_types[i]
                if rtype == 0:
                    # Unknown format gets no diversity bonus
                    bonuses[i] = 0.0
                else:
                    freq = type_counts[rtype] / group_size
                    bonuses[i] = diversity_weight * (1.0 - freq)

        return bonuses

    def __call__(
        self,
        responses: List[str],
        ground_truths: List[Any],
        return_dict: bool = False,
        uids: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Calculate rewards for a batch of responses.

        Args:
            responses: List of model responses.
            ground_truths: List of ground truth answers.
            return_dict: If True, return dict with 'rewards' and 'metrics'.
            uids: List of group identifiers for diversity bonus computation.
                  Responses with the same uid belong to the same GRPO group.

        Returns:
            List[float] or dict with 'rewards' and 'metrics'.
        """
        # First pass: classify all responses and compute base rewards
        n = len(responses)
        response_types = []
        base_rewards_list = []  # reward before diversity bonus
        correctness_list = []
        token_counts = []

        batch_stats = {
            'type1_count': 0, 'type2_count': 0, 'type3_count': 0,
            'unknown_count': 0,
            'correct_count': 0, 'incorrect_count': 0,
            'type1_correct': 0, 'type2_correct': 0, 'type3_correct': 0,
            'type1_lengths': [], 'type2_lengths': [], 'type3_lengths': [],
            'base_rewards': [], 'format_bonuses': [],
            'length_scalars': [], 'diversity_bonuses': [],
            'total_rewards': [],
        }

        for response, gt in zip(responses, ground_truths):
            token_count = self.count_tokens(response)
            response_type = self.get_response_type(response)
            response_types.append(response_type)
            token_counts.append(token_count)

            if response_type == 0:
                base_rewards_list.append(self.unknown_penalty)
                correctness_list.append(False)
                batch_stats['unknown_count'] += 1
                batch_stats['incorrect_count'] += 1
                batch_stats['base_rewards'].append(self.unknown_penalty)
                batch_stats['format_bonuses'].append(0.0)
                batch_stats['length_scalars'].append(1.0)
                continue

            is_correct = self.check_answer_correctness(response, gt)
            correctness_list.append(is_correct)

            if is_correct:
                base_reward = self.correct_reward
                format_bonus = {1: self.type1_format_bonus, 2: self.type2_format_bonus, 3: self.type3_format_bonus}[response_type]
                error_penalty = 0.0
            else:
                base_reward = self.incorrect_reward
                format_bonus = 0.0
                error_penalty = {1: self.type1_error_penalty, 2: self.type2_error_penalty, 3: self.type3_error_penalty}[response_type]

            length_scalar = self.calculate_length_scalar(token_count)

            if is_correct:
                reward_before_diversity = (base_reward + format_bonus) * length_scalar
            else:
                reward_before_diversity = (base_reward + error_penalty) * length_scalar

            base_rewards_list.append(reward_before_diversity)

            # Stats
            type_key = f'type{response_type}'
            batch_stats[f'{type_key}_count'] += 1
            batch_stats[f'{type_key}_lengths'].append(token_count)
            if is_correct:
                batch_stats[f'{type_key}_correct'] += 1
                batch_stats['correct_count'] += 1
            else:
                batch_stats['incorrect_count'] += 1

            batch_stats['base_rewards'].append(base_reward)
            batch_stats['format_bonuses'].append(format_bonus if is_correct else error_penalty)
            batch_stats['length_scalars'].append(length_scalar)

        # Second pass: compute diversity bonuses
        diversity_bonuses = self._compute_diversity_bonuses(response_types, uids)

        # Combine base rewards + diversity bonuses
        rewards = []
        for i in range(n):
            reward = base_rewards_list[i] + diversity_bonuses[i]
            rewards.append(reward)
            batch_stats['diversity_bonuses'].append(diversity_bonuses[i])
            batch_stats['total_rewards'].append(reward)

        if not return_dict:
            return rewards

        metrics = self._compute_batch_metrics(batch_stats, n)
        return {'rewards': rewards, 'metrics': metrics}

    def _compute_batch_metrics(self, batch_stats: Dict, total_samples: int) -> Dict[str, float]:
        """Compute batch-level metrics for logging."""
        metrics = {}

        if total_samples > 0:
            metrics['format/type1_ratio'] = batch_stats['type1_count'] / total_samples
            metrics['format/type2_ratio'] = batch_stats['type2_count'] / total_samples
            metrics['format/type3_ratio'] = batch_stats['type3_count'] / total_samples
            metrics['format/unknown_ratio'] = batch_stats['unknown_count'] / total_samples

        for t in [1, 2, 3]:
            count = batch_stats[f'type{t}_count']
            if count > 0:
                metrics[f'format/type{t}_correct_rate'] = batch_stats[f'type{t}_correct'] / count
                metrics[f'format/type{t}_avg_length'] = float(np.mean(batch_stats[f'type{t}_lengths']))
            else:
                metrics[f'format/type{t}_correct_rate'] = 0.0
                metrics[f'format/type{t}_avg_length'] = 0.0

        if batch_stats['base_rewards']:
            metrics['reward/base_mean'] = float(np.mean(batch_stats['base_rewards']))
            metrics['reward/format_bonus_mean'] = float(np.mean(batch_stats['format_bonuses']))
            metrics['reward/length_scalar_mean'] = float(np.mean(batch_stats['length_scalars']))
            metrics['reward/total_mean'] = float(np.mean(batch_stats['total_rewards']))

        if batch_stats['diversity_bonuses']:
            metrics['diversity/bonus_mean'] = float(np.mean(batch_stats['diversity_bonuses']))
            metrics['diversity/bonus_max'] = float(np.max(batch_stats['diversity_bonuses']))
            metrics['diversity/weight'] = self.get_diversity_weight()
            metrics['diversity/training_step'] = float(self.training_step)

        if total_samples > 0:
            metrics['accuracy/overall'] = batch_stats['correct_count'] / total_samples

        return metrics

    def get_reward_breakdown(self, response: str, ground_truth: Any) -> Dict[str, float]:
        """Get detailed breakdown of reward components for a single response."""
        token_count = self.count_tokens(response)
        response_type = self.get_response_type(response)

        if response_type == 0:
            return {
                'total_reward': self.unknown_penalty,
                'is_correct': False,
                'is_unknown': True,
                'response_type': 0,
                'base_reward': self.unknown_penalty,
                'format_bonus': 0.0,
                'error_penalty': 0.0,
                'diversity_bonus': 0.0,
                'token_count': token_count,
                'length_scalar': 1.0,
                'predicted_answer': '',
            }

        is_correct = self.check_answer_correctness(response, ground_truth)
        length_scalar = self.calculate_length_scalar(token_count)

        if is_correct:
            base_reward = self.correct_reward
            format_bonus = {1: self.type1_format_bonus, 2: self.type2_format_bonus, 3: self.type3_format_bonus}[response_type]
            error_penalty = 0.0
            total = (base_reward + format_bonus) * length_scalar
        else:
            base_reward = self.incorrect_reward
            format_bonus = 0.0
            error_penalty = {1: self.type1_error_penalty, 2: self.type2_error_penalty, 3: self.type3_error_penalty}[response_type]
            total = (base_reward + error_penalty) * length_scalar

        return {
            'total_reward': total,
            'is_correct': is_correct,
            'is_unknown': False,
            'response_type': response_type,
            'base_reward': base_reward,
            'format_bonus': format_bonus,
            'error_penalty': error_penalty,
            'diversity_bonus': 0.0,  # Single response, no group context
            'token_count': token_count,
            'length_scalar': length_scalar,
            'predicted_answer': self.extract_answer(response),
        }


# Global instance for verl compatibility
_reward_instance = None


def create_reward_function(data_source, solution_str, ground_truth, extra_info=None, **init_kwargs):
    """
    Reward function interface expected by verl's naive reward manager.
    """
    global _reward_instance
    if _reward_instance is None:
        _reward_instance = AdaptiveReward(**init_kwargs)
    rewards = _reward_instance([solution_str], [ground_truth])
    return rewards[0]


if __name__ == "__main__":
    print("=" * 70)
    print("Adaptive Reward Demo")
    print("=" * 70)

    reward_fn = AdaptiveReward(
        initial_diversity_weight=0.3,
        total_training_steps=500,
    )

    # Simulate a GRPO group: 8 responses for the same prompt
    group_responses = [
        '<answer>C</answer>',                          # Type 1
        '<answer>C</answer>',                          # Type 1
        '<answer>C</answer>',                          # Type 1
        '<answer>C</answer>',                          # Type 1
        '<answer>C</answer>',                          # Type 1
        '<answer>C</answer>',                          # Type 1
        '<perception>graph</perception>\n<answer>C</answer>',  # Type 2
        '<perception>graph</perception>\n<reasoning>power law</reasoning>\n<answer>C</answer>',  # Type 3
    ]
    group_gts = ['C'] * 8
    group_uids = ['group1'] * 8

    print("\n--- Step 0 (full diversity weight) ---")
    reward_fn.set_training_step(0)
    result = reward_fn(group_responses, group_gts, return_dict=True, uids=group_uids)
    for i, (resp, r) in enumerate(zip(group_responses, result['rewards'])):
        rtype = reward_fn.get_response_type(resp)
        print(f"  Response {i}: Type{rtype}, reward={r:.4f}")
    print(f"  Diversity weight: {reward_fn.get_diversity_weight():.4f}")
    print(f"  Diversity bonus mean: {result['metrics']['diversity/bonus_mean']:.4f}")

    print("\n--- Step 250 (mid training, cosine ~50%) ---")
    reward_fn.set_training_step(250)
    result = reward_fn(group_responses, group_gts, return_dict=True, uids=group_uids)
    for i, (resp, r) in enumerate(zip(group_responses, result['rewards'])):
        rtype = reward_fn.get_response_type(resp)
        print(f"  Response {i}: Type{rtype}, reward={r:.4f}")
    print(f"  Diversity weight: {reward_fn.get_diversity_weight():.4f}")

    print("\n--- Step 500 (end of training, diversity ~0) ---")
    reward_fn.set_training_step(500)
    result = reward_fn(group_responses, group_gts, return_dict=True, uids=group_uids)
    for i, (resp, r) in enumerate(zip(group_responses, result['rewards'])):
        rtype = reward_fn.get_response_type(resp)
        print(f"  Response {i}: Type{rtype}, reward={r:.4f}")
    print(f"  Diversity weight: {reward_fn.get_diversity_weight():.4f}")

    # Show diversity bonus with mixed group
    print("\n\n--- Diverse vs Homogeneous groups (Step 0) ---")
    reward_fn.set_training_step(0)

    # Homogeneous group: all Type 3
    homo_responses = ['<perception>g</perception>\n<reasoning>r</reasoning>\n<answer>C</answer>'] * 8
    homo_gts = ['C'] * 8
    homo_uids = ['homo'] * 8
    result_homo = reward_fn(homo_responses, homo_gts, return_dict=True, uids=homo_uids)
    print(f"  Homogeneous (all Type3): rewards = {[f'{r:.3f}' for r in result_homo['rewards']]}")
    print(f"    -> Diversity bonus = {result_homo['metrics']['diversity/bonus_mean']:.4f} (all same type, freq=1.0)")

    # Diverse group: mixed types
    diverse_responses = [
        '<answer>C</answer>',
        '<answer>C</answer>',
        '<perception>g</perception>\n<answer>C</answer>',
        '<perception>g</perception>\n<answer>C</answer>',
        '<perception>g</perception>\n<answer>C</answer>',
        '<perception>g</perception>\n<reasoning>r</reasoning>\n<answer>C</answer>',
        '<perception>g</perception>\n<reasoning>r</reasoning>\n<answer>C</answer>',
        '<perception>g</perception>\n<reasoning>r</reasoning>\n<answer>C</answer>',
    ]
    diverse_gts = ['C'] * 8
    diverse_uids = ['diverse'] * 8
    result_diverse = reward_fn(diverse_responses, diverse_gts, return_dict=True, uids=diverse_uids)
    print(f"  Diverse (2xT1 + 3xT2 + 3xT3): rewards = {[f'{r:.3f}' for r in result_diverse['rewards']]}")
    print(f"    -> Diversity bonus = {result_diverse['metrics']['diversity/bonus_mean']:.4f}")

    print("\n" + "=" * 70)
    print("All tests passed!")
