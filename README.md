# robustness
A central place to test model agnositc local interpretable explanation models (taken from following libraries):

* [LIME](https://github.com/marcotcr/lime) (For LIME),
* [Anchor](https://github.com/SeldonIO/alibi) (For Anchor),
* [DeepExplain](https://github.com/marcoancona/DeepExplain) (for gradient based methods)

And give an empirical notion of robustness. (currently maintained for image dataType only)


### Dependencies

The dependecies of above libraries.


### Use cases

```
imagenet.py (Give the explanation and robustness of different methods on imagenet dataset, trained on InceptionV3)
mnist.py (Give the explanation and robustness of different methods on imagenet dataset, trained on NN from sample_nn.py)

```
