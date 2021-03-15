# Benchmarks

The benchmark script was heavily inspired by [this](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) Pytorch recipe for AMP. I ran the script on a Quadro P5000 GPU, with the following parameters:

- sequence length: 100
- batch size: 100
- number of tags: 10
- number of tags: 1000

As you can see in the script, no masking or impossible transitions are used, inputs are passed with batch dimension first, and the model parameters are set up to be the same for each module. The packages I tested are:

- this one, obviously
- [torchcrf](https://github.com/kmkurn/pytorch-crf) by kmkurn
- [pytorchcrf](https://github.com/statech/pytorchCRF) by statech
- [TorchCRF](https://github.com/s14t284/TorchCRF) by s14t284

I chose these packages because they are the top 3 results on PyPi for the query "torch crf", if you filter for the MIT license. The model APIs are similar, although the `pytorchcrf` as a small advantage for decoding, as it returns tensors instead of lists during decoding (meaning it avoid the conversion to truncated lists, depending on the mask). I averaged the results over 5 tries and report the mean and standard deviation for the total time (the maximum memory usage stayed the same across all runs).

## Forward + backward pass


| package      | execution time mean | execution time std | max memory usage |
|--------------|---------------------|--------------------|------------------|
| `linear_crf` | **54.046 sec**      | 0.557 sec          | 4.766 MB         |
| `torchcrf`   | 104.527 sec         | 0.962 sec          | 4.811 MB         |
| `pytorchcrf` | 104.143 sec         | 0.880 sec          | 4.805 MB         |
| `TorchCRF`   | 105.101 sec         | 0.532 sec          | 4.770 MB         |

As you can see, my package is about twice as fast, for similar memory consumption (actually a tiny bit less). There aren't really any differences between the other packages, as they all have about the same execution times and peak memory usage.

## Viterbi decoding

| package                    | execution time mean | execution time std | max memory usage |
|----------------------------|---------------------|--------------------|------------------|
| `linear_crf`               | **33.056 sec**      | 0.713 sec          | 1.327 MB         |
| `linear_crf` tensor output | **25.945 sec**      | 0.391 sec          | 1.327 MB         |
| `torchcrf`                 | 207.218 sec         | 0.825 sec          | 1.267 MB         |
| `pytorchcrf`               | **24.224 sec**      | 0.405 sec          | 2.035 MB         |
| `TorchCRF`                 | 222.264 sec         | 1.336 sec          | 1.670 MB         |

For the decoding results are not as one sided, with two distinct groups emerging. It appears that, although it comes with about a peak memory usage 50 % greater, `pytorchcrf` comes out first on the Viterbi decoding. As I was curious to see the importance of the gain `pytorchcrf`  got from directly returning the tensor I modified my package to return the `best_tags` as a tensor, skipping the last line of code. Although it still lags behind, the difference in that case is greatly reduced, for a much smaller memory footprint.

In all cases, the performance gap between the two groups is quite notable. `torchcrf` and `TorchCRF` are completely outperformed on execution time, with not that much to show for with regards to memory usage. As I developped this project using `torchcrf` as a starting point I wasn't that surprised, as I was aware that the decoding was not vectorized with regards to the batch dimension - I imagine that is the case for `TorchCRF` as well.

In order to confirm the trend I did two more decoding benchmarks, simply modifying these parameters:

- batch size 20 and sequence length 200
- batch size 200 and sequence length 20

The results can be found in the tables below (results are similar for the forward / backward pass so I didn't inclue them):

| package      | execution time mean | execution time std | max memory usage |
|--------------|---------------------|--------------------|------------------|
| `linear_crf` | **51.238 sec**      | 0.343 sec          | 0.532 MB         |
| `torchcrf`   | 105.041 sec         | 0.921 sec          | 0.591 MB         |
| `pytorchcrf` | **49.025 sec**      | 0.353 sec          | 0.813 MB         |
| `TorchCRF`   | 103.431 sec         | 0.439 sec          | 0.770 MB         |

| package      | execution time mean | execution time std | max memory usage |
|--------------|---------------------|--------------------|------------------|
| `linear_crf` | 20.370 sec          | 0.311 sec          | 0.635 MB         |
| `torchcrf`   | 94.029 sec          | 2.263 sec          | 0.619 MB         |
| `pytorchcrf` | **5.173 sec**       | 0.106 sec          | 0.872 MB         |
| `TorchCRF`   | 112.748 sec         | 2.526 sec          | 0.828 MB         |

As you can see, the gap is definitely due to the lack of vectorization. It also become more clear that the difference between this repository and `pytorchcrf` is almost entirely due to the transition from tensor to list, which widens as the batch grows in size.
