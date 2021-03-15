from typing import List
from typing import Optional

import torch
from torch import BoolTensor
from torch import FloatTensor
from torch import LongTensor
import torch.nn as nn


class LinearCRF(nn.Module):
    def __init__(
        self,
        num_tags: int,
        batch_first: bool = False,
        impossible_starts: Optional[BoolTensor] = None,
        impossible_transitions: Optional[BoolTensor] = None,
        impossible_ends: Optional[BoolTensor] = None,
    ):
        super(LinearCRF, self).__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.starts = nn.Parameter(torch.empty(self.num_tags))
        self.transitions = nn.Parameter(torch.empty(self.num_tags, self.num_tags))
        self.ends = nn.Parameter(torch.empty(self.num_tags))
        self.reset_parameters(impossible_starts, impossible_transitions, impossible_ends)

    def reset_parameters(
        self,
        impossible_starts: Optional[BoolTensor] = None,
        impossible_transitions: Optional[BoolTensor] = None,
        impossible_ends: Optional[BoolTensor] = None,
    ) -> None:
        """
        Initialize the parameters of the model, impossible starts / transition / ends are set to
        -10000 to avoid being considered.

        Parameters:
        -----------
            impossible_starts: Optional[BoolTensor]
                shape: (num_tags,)
                impossible starting tags
            impossible_transitions: Optional[BoolTensor]
                shape: (num_tags, num_tags)
                impossible transition ([i,j] = True means tag i -> tag j is impossible)
            impossible_ends: Optional[BoolTensor]
                shape: (num_tags,)
                impossible ending tags
        """
        for param, impossible in zip(
            self.parameters(), [impossible_starts, impossible_transitions, impossible_ends]
        ):
            nn.init.uniform_(param, -0.1, 0.1)
            if impossible is not None:
                with torch.no_grad():
                    param.masked_fill_(impossible, -10000)

    def forward(
        self, emissions: FloatTensor, labels: LongTensor, mask: Optional[BoolTensor] = None
    ) -> FloatTensor:
        """
        Computes the negative log-likelihood given emission scores for a sequence of tags using the
        forward algorithm.

        Parameters:
        -----------
            emissions: FloatTensor
                shape: (seq_length, batch_size, num_tags) if batch_first is False
                emission score for each tag type and timestep
            labels: LongTensor
                shape: (seq_length, batch_size) if batch_first is False
                ground truth tag sequences
            mask: Optional[BoolTensor]
                shape: (seq_length, batch_size) if batch_first is False
                optional boolean mask for each sequence

        Returns:
        --------
            result: torch.FloatTensor
                shape: ()
                Negative log-likelihood normalized by the mask sum
        """
        if mask is None:
            mask = torch.ones_like(labels, dtype=torch.bool)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            labels = labels.transpose(0, 1)
            mask = mask.transpose(0, 1)

        numerator = self.starts[labels[0]]
        numerator += (emissions.gather(2, labels.unsqueeze(-1)).squeeze(-1) * mask).sum(dim=0)
        numerator += (self.transitions[labels[:-1], labels[1:]] * mask[1:]).sum(dim=0)
        seq_ends = mask.long().sum(dim=0) - 1
        last_tags = labels.gather(0, seq_ends.unsqueeze(0)).squeeze(0)
        numerator += self.ends[last_tags]

        denominator = self.starts + emissions[0]
        broadcast_emissions = emissions.unsqueeze(2)

        for i in range(1, labels.shape[0]):
            broadcast_denominator = denominator.unsqueeze(2)
            next_denominator = broadcast_denominator + self.transitions + broadcast_emissions[i]
            next_denominator = next_denominator.logsumexp(dim=1)
            denominator = next_denominator.where(mask[i].unsqueeze(1), denominator)

        denominator += self.ends
        denominator = denominator.logsumexp(dim=1)

        llh = numerator - denominator
        return -llh.sum() / mask.sum()

    @torch.no_grad()
    def decode(self, emissions: FloatTensor, mask: Optional[BoolTensor] = None) -> List[List[int]]:
        """
        Computes the best tag sequence given emission scores using the Viterbi algorithm.

        Parameters:
        -----------
            emissions: FloatTensor
                shape: (seq_length, batch_size, num_tags) if batch_first is False
                emission score for each tag type and timestep
            mask: Optional[BoolTensor]
                shape: (seq_length, batch_size) if batch_first is False
                optional boolean mask for each sequence

        Returns:
        --------
            best_tags: List[List[int]]
                shape: (batch_size, real seq_length)
                best tag sequences
        """

        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.bool)
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        seq_length, batch_size = mask.shape
        broadcast_emissions = emissions.unsqueeze(2)
        score = self.starts + emissions[0]
        history = emissions.new_empty(seq_length - 1, batch_size, self.num_tags, dtype=torch.long)

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            next_score = broadcast_score + self.transitions + broadcast_emissions[i]
            next_score, next_indices = next_score.max(dim=1)
            score = next_score.where(mask[i].unsqueeze(1), score)
            history[-i] = next_indices.where(mask[i - 1].unsqueeze(1), history[-i + 1])

        score += self.ends
        best_tags = torch.empty_like(mask, dtype=torch.long)
        _, best_last_tag = score.max(dim=1)
        best_prev_tag = best_last_tag

        for i in range(seq_length - 1):
            best_prev_tag = history[i].gather(1, best_prev_tag.unsqueeze(-1)).squeeze(-1)
            best_tags[seq_length - 2 - i] = best_prev_tag

        seq_ends = mask.long().sum(dim=0) - 1
        best_tags.scatter_(0, seq_ends.unsqueeze(0), best_last_tag.unsqueeze(0))
        return [line[: end + 1].tolist() for line, end in zip(best_tags.t(), seq_ends)]
