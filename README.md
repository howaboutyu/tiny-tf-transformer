## tiny-tf-transformer (yet another transformer implementation)

This is a tiny transformer implementation in tensorflow 2. It is based on the [attention is all you need](https://arxiv.org/abs/1706.03762) paper. The transformer implementation is based on the [tensorflow tutorial](https://www.tensorflow.org/tutorials/text/transformer). It is mainly used for my own learning purposes and also has some paper implementation related to transformers.


### Installation

```bash
pip install .
```

### Projects

- [x] [Transformer](https://arxiv.org/abs/1706.03762)
- [x] [Pix2Seq](https://arxiv.org/abs/2109.10852): A Language Modeling Framework for Object Detection. [Code](examples/pix2seq)
- [ ] [ViLT](https://arxiv.org/abs/2102.03334): Vision-and-Language Transformer Without Convolution or Region Supervision. 
- [ ] [BART](https://arxiv.org/abs/1910.13461): Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.
- [ ] [T5](https://arxiv.org/abs/1910.10683): Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer.

### Tests

```bash
pytest
```

### References

- [Attention is all you need](https://arxiv.org/abs/1706.03762)
- [Transformer tutorial](https://www.tensorflow.org/tutorials/text/transformer)
