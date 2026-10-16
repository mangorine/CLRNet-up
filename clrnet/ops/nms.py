# Copyright (c) 2018, Grégoire Payen de La Garanderie, Durham University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch


def lane_overlap(boxes, thresh):
    n_offsets = boxes.shape[1] - 5
    n_strips = n_offsets - 1
    lengths = boxes[:, 4]
    starts = (boxes[:, 2] * n_strips + 0.5).long()
    ends = (starts + lengths - 1 + 0.5 - (lengths - 1 < 0).to(boxes.dtype)).long()

    start = torch.max(starts[:, None], starts[None, :])
    end = torch.min(torch.min(ends[:, None], ends[None, :]),
                    ends.new_tensor(n_offsets - 1))

    offsets = boxes[:, 5:]
    r = torch.arange(n_offsets, device=boxes.device)
    valid = (r >= start[..., None]) & (r <= end[..., None])
    dist = ((offsets[:, None, :] - offsets[None, :, :]).abs() * valid).sum(-1)

    overlap = (end >= start) & (dist < thresh * (end - start + 1))
    return overlap.triu(diagonal=1)


def nms(boxes, scores, overlap, top_k):
    num_boxes = boxes.shape[0]
    device = boxes.device
    idx = scores.argsort(descending=True)
    mask = lane_overlap(boxes[idx], overlap).cpu()
    idx_cpu = idx.cpu()

    keep = torch.zeros(num_boxes, dtype=torch.long)
    parent_object_index = torch.zeros(num_boxes, dtype=torch.long)
    removed = torch.zeros(num_boxes, dtype=torch.bool)
    num_to_keep = 0
    for i in range(num_boxes):
        if removed[i]:
            continue
        keep[num_to_keep] = idx_cpu[i]
        removed |= mask[i]
        num_to_keep += 1
        parent_object_index[idx_cpu[mask[i]]] = num_to_keep
        parent_object_index[idx_cpu[i]] = num_to_keep
        if num_to_keep == top_k:
            break

    return (keep.to(device), torch.tensor(num_to_keep, device=device),
            parent_object_index.to(device))
