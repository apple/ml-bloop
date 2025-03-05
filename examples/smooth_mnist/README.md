# Smooth MLP for MNIST classification

In this example, we aim to train an MLP for MNIST classification while keeping the model smooth with an auxiliary loss that measures the smoothness of the network.
This is a simplified version of the first experiment introduced in Section 5.2 of our [paper](https://arxiv.org/abs/2402.02998).

Concretely, the main loss is the cross-entropy loss while the auxiliary loss is the logarithm of an upper bound on the Lipchitz constant of the network.
$$L_{\text{aux}} = \log\big(\prod_{l=1}^{L}\parallel W_l\parallel_2\big)$$
Here, $W_l$ is the weight matrix of the $l$-th layer of the network.


## Additional dependencies

To run this example, please install additional dependencies with

```
pip install -r requirements.txt
```


## Training

Our main training script is `train_smooth_mnist.py`.

```
python train_smooth_mnist.py
```

This performs training across the four gradient combining methods `mixed`, `bloop`, `pcgrad`, and `dynamic_barrier`, with lambda values ranging from 1e-4 to 1 in logspace. The resulting metrics are saved to `results/metrics`.

You can change the included methods and their corresponding emas using `--methods` and `--emas`, and change the metric directory with `--metric_dir` (some other hyperparameters can be changed as well, please refer to the script for full details).

```
python train_smooth_mnist.py \
    --methods bloop pcgrad \
    --emas 0.001 0.1 \
    --metric_dir results/metrics_bloop_pcgrad
```


## Results

We visualize the Pareto front of the two losses with

```
python plot_smooth_mnist.py
```

From the results below, we see that BLOOP allows us to obtain a better Pareto front compared to other methods. 
Dynamic barrier and naive mixing of gradients perform closely to each other, while PCGrad exhibits a slightly different behavior due to double projections.

<p align="center">
  <img src="results/figures/pareto_train_loss_aux_loss.png" alt="Pareto train" width="400"/>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="results/figures/pareto_test_loss_aux_loss.png" alt="Pareto test" width="400"/>
</p>
