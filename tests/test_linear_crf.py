import pytest
import torch
from torch.optim import SGD

from linear_crf import LinearCRF


@pytest.mark.parametrize("n_tags", range(2, 20))
def test_init_default(n_tags):
    model = LinearCRF(n_tags)
    assert model.starts.shape == (n_tags,)
    assert model.transitions.shape == (n_tags, n_tags)
    assert model.ends.shape == (n_tags,)
    assert model.starts.dtype == torch.float32
    assert model.transitions.dtype == torch.float32
    assert model.ends.dtype == torch.float32
    assert (-0.1 <= model.starts).all() and (model.starts <= 0.1).all()
    assert (-0.1 <= model.transitions).all() and (model.transitions <= 0.1).all()
    assert (-0.1 <= model.ends).all() and (model.ends <= 0.1).all()
    for param in model.parameters():
        assert param.grad is None
    assert isinstance(model.batch_first, bool) and not model.batch_first


@pytest.mark.parametrize("batch_first", [False, True])
def test_init_batch_first(batch_first):
    n_tags = 10
    model = LinearCRF(n_tags, batch_first=batch_first)
    assert model.batch_first == batch_first


@pytest.mark.parametrize(
    "impossible_starts", [torch.randint(2, (10,), dtype=torch.bool) for _ in range(10)]
)
def test_init_impossible_starts(impossible_starts):
    n_tags = 10
    model = LinearCRF(n_tags, impossible_starts=impossible_starts)
    assert (model.starts[impossible_starts] == -10000).all()
    assert (-0.1 <= model.starts[~impossible_starts]).all() and (
        model.starts[~impossible_starts] <= 0.1
    ).all()
    assert (-0.1 <= model.transitions).all() and (model.transitions <= 0.1).all()
    assert (-0.1 <= model.ends).all() and (model.ends <= 0.1).all()
    for param in model.parameters():
        assert param.grad is None


@pytest.mark.parametrize(
    "impossible_transitions", [torch.randint(2, (10, 10), dtype=torch.bool) for _ in range(10)]
)
def test_init_impossible_transitions(impossible_transitions):
    n_tags = 10
    model = LinearCRF(n_tags, impossible_transitions=impossible_transitions)
    assert (model.transitions[impossible_transitions] == -10000).all()
    assert (-0.1 <= model.transitions[~impossible_transitions]).all() and (
        model.transitions[~impossible_transitions] <= 0.1
    ).all()
    assert (-0.1 <= model.starts).all() and (model.starts <= 0.1).all()
    assert (-0.1 <= model.ends).all() and (model.ends <= 0.1).all()
    for param in model.parameters():
        assert param.grad is None


@pytest.mark.parametrize(
    "impossible_ends", [torch.randint(2, (10,), dtype=torch.bool) for _ in range(10)]
)
def test_init_impossible_ends(impossible_ends):
    n_tags = 10
    impossible_ends = torch.randint(2, (10,), dtype=torch.bool)
    model = LinearCRF(n_tags, impossible_ends=impossible_ends)
    assert (model.ends[impossible_ends] == -10000).all()
    assert (-0.1 <= model.ends[~impossible_ends]).all() and (
        model.ends[~impossible_ends] <= 0.1
    ).all()
    assert (-0.1 <= model.starts).all() and (model.starts <= 0.1).all()
    assert (-0.1 <= model.transitions).all() and (model.transitions <= 0.1).all()
    for param in model.parameters():
        assert param.grad is None


def test_reset_parameters():
    n_tags = 10
    model = LinearCRF(n_tags)
    starts = model.starts.clone().detach()
    transitions = model.transitions.clone().detach()
    ends = model.ends.clone().detach()
    model.reset_parameters()
    assert (model.starts != starts).all()
    assert (model.transitions != transitions).all()
    assert (model.ends != ends).all()
    for param in model.parameters():
        assert param.grad is None


def test_forward():
    n_tags = 10
    seq_length = 3
    batch_size = 4
    model = LinearCRF(n_tags)
    emissions = torch.randn(seq_length, batch_size, n_tags)
    labels_1 = torch.randint(n_tags, (seq_length, batch_size))
    labels_2 = torch.randint(n_tags, (seq_length, batch_size))
    if (labels_1 == labels_2).all():
        labels_1[0, 0] = n_tags - labels_2[0, 0] - 1
    loss_1 = model(emissions, labels_1)
    loss_2 = model(emissions, labels_2)
    assert loss_1 != loss_2


def test_forward_half_precision():
    n_tags = 10
    seq_length = 3
    batch_size = 4
    model = LinearCRF(n_tags)
    emissions = torch.randn(seq_length, batch_size, n_tags, dtype=torch.float16)
    labels = torch.randint(n_tags, (seq_length, batch_size))
    model(emissions, labels)


def test_forward_backward():
    n_tags = 10
    seq_length = 3
    batch_size = 4
    model = LinearCRF(n_tags)
    opt = SGD(model.parameters(), lr=0.1)
    emissions = torch.randn(seq_length, batch_size, n_tags)
    labels = torch.randint(n_tags, (seq_length, batch_size))
    loss = model(emissions, labels)
    loss.backward()
    for param in model.parameters():
        assert param.grad is not None
    opt.step()
    new_loss = model(emissions, labels)
    assert new_loss < loss


@pytest.mark.parametrize("seq_length", [2, 3, 4])
def test_forward_batch_first(seq_length):
    n_tags = 10
    batch_size = 3
    model = LinearCRF(n_tags)
    emissions = torch.randn(seq_length, batch_size, n_tags)
    labels = torch.randint(n_tags, (seq_length, batch_size))
    loss_1 = model(emissions, labels)
    model.batch_first = True
    loss_2 = model(emissions.transpose(0, 1), labels.t())
    assert loss_1 == loss_2
    loss_3 = model(emissions, labels)
    assert loss_1 != loss_3


def test_forward_mask():
    n_tags = 10
    seq_length = 3
    batch_size = 4
    model = LinearCRF(n_tags)
    emissions = torch.randn(seq_length, batch_size, n_tags)
    labels = torch.randint(n_tags, (seq_length, batch_size))
    mask = torch.ones_like(labels, dtype=torch.bool)
    loss_1 = model(emissions, labels)
    loss_2 = model(emissions, labels, mask=mask)
    assert loss_1 == loss_2
    mask[-1, 0] = False
    loss_3 = model(emissions, labels, mask=mask)
    assert loss_1 != loss_3


def test_decode():
    n_tags = 10
    seq_length = 3
    batch_size = 4
    model = LinearCRF(n_tags)
    emissions_1 = torch.randn(seq_length, batch_size, n_tags)
    emissions_2 = torch.randn(seq_length, batch_size, n_tags)
    y_1 = model.decode(emissions_1)
    y_2 = model.decode(emissions_2)
    assert y_1 != y_2
    for y in [y_1, y_2]:
        assert len(y) == batch_size
        for seq in y:
            assert len(seq) == seq_length
            assert all(0 <= e <= n_tags for e in seq)


def test_decode_half_precision():
    n_tags = 10
    seq_length = 3
    batch_size = 4
    model = LinearCRF(n_tags)
    emissions = torch.randn(seq_length, batch_size, n_tags, dtype=torch.float16)
    model.decode(emissions)


@pytest.mark.parametrize("seq_length", [2, 3, 4])
def test_decode_batch_first(seq_length):
    n_tags = 10
    batch_size = 3
    model = LinearCRF(n_tags)
    emissions = torch.randn(seq_length, batch_size, n_tags)
    y_1 = model.decode(emissions)
    model.batch_first = True
    y_2 = model.decode(emissions.transpose(0, 1))
    assert y_1 == y_2
    y_3 = model.decode(emissions)
    assert y_1 != y_3
    for y in [y_1, y_2]:
        assert len(y) == batch_size
        for seq in y:
            assert len(seq) == seq_length
            assert all(0 <= e <= n_tags for e in seq)
    assert len(y_3) == seq_length
    for seq in y_3:
        assert len(seq) == batch_size
        assert all(0 <= e <= n_tags for e in seq)


def test_decode_mask():
    n_tags = 10
    seq_length = 3
    batch_size = 4
    model = LinearCRF(n_tags)
    emissions = torch.randn(seq_length, batch_size, n_tags)
    mask = torch.ones_like(emissions[:, :, 0], dtype=torch.bool)
    y_1 = model.decode(emissions)
    y_2 = model.decode(emissions, mask=mask)
    assert y_1 == y_2
    mask[-1, 0] = False
    y_3 = model.decode(emissions, mask=mask)
    assert y_1 != y_3
    assert y_1[1:] == y_3[1:]
    assert len(y_3[0]) == seq_length - 1
