# Linear CRF

## Description

This repository hosts my Pytorch implementation of the Linear Conditional Random Field model, which is available via PyPi. There are a number of similar packages, however at this point I can say this one is faster (or equivalent), as I benchmarked the alternatives.

## Origin story

About a year ago I needed this for a project I was working on, and stumbled upon the de facto _official_ implementation (at least by star count), which you can check [here](https://github.com/kmkurn/pytorch-crf). Now although the code was already quite good I decided to optimize it for my needs, and ended up with significant gains. I decided to open a PR in order to share, but quickly realized two thjings. The first was that I had commited way too much, and should have consulted with the author first. The second was that some changes actually went against the wants of the author, as they didn't meet his readability standard. Anyway, long story short I got demotivated, then forgot about it, and the [PR of shame](https://github.com/kmkurn/pytorch-crf/pull/54) is still open >_<"

However the story does not stop here ! As I needed to use a CRF for another project recently, I decided to clean up my code, and ended up optimizing it even further. As I like how it looks, but learned my lesson, I decided to release it, and _voila_ !

## Installation

With Python 3.6 or higher:

```
pip install linear-crf-torch
```

The model is not compatible with Pytorch versions older then 1.3, as I use features added from that version. The required changes are minimal, si I don't plan to include them.

## Usage

The example below shows the basic usage: 

```python
import torch
from linear_crf import LinearCRF

seq_length = 3
batch_size = 4
num_tags = 5

model = LinearCRF(num_tags)

emissions = torch.randn(seq_length, batch_size, num_tags)
labels = torch.randint(num_tags, (seq_length, batch_size))

# Compute the average negative log-likelihood
loss = model(emissions, labels)
print(f"loss: {loss:.4f}")

# Viterbi decoding
tags = model.decode(emissions)
for i, x in enumerate(tags):
    print(f"tags for sequence {i}: {x}") 
```

A couple caveats:

- I followed the Pytorch convention of setting the batch dimension after the sequence one, but you can set `batch_first=True` in the constructor if you wish to pass data the other way around.
- Unlike similar packages, no input validation is performed - I think the documentation should be enough to avoid any bugs.
- Using the `impossible_starts`, `impossible_transitions` and `impossible_ends` parameters in the constructor, you can make it impossible for certain tags to appear at the start or end of the sequences, and make transitions from one tag to another impossible.
- In the forward pass, the loss is normalized by the number of non-masked elements. It doesn't make sense to normalize in an other way, neither does directly using the sum.
- Gradients are disabled during decoding.
- Masking is only supported from the right, meaning if you mask the left part of a sentence (e.g. `[0, 0, 1, 1]`) the computations will be incorrect.

## License

[MIT](LICENSE)

## Benchmarks

Over [here](benchmarks/README.md).

## Contributing

All help is welcome, as long as you open an issue beforehand to talk about it :)

## Reference

[Conditional Random Fields: Probabilistic Modelsfor Segmenting and Labeling Sequence Data](https://repository.upenn.edu/cgi/viewcontent.cgi?article=1162&context=cis_papers) by John Lafferty, Andrew McCallum and Fernando C.N. Pereira
