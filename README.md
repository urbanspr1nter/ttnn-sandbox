# tt-sandbox

## What is this repo?

This repo is a sandbox for experimenting with the Tenstorrent TTNN library.

It was meant as a place to just dump a bunch of notebooks and teach myself how to use the Tensix hardware. 

I have a Wormhole n150d, so I created this repo in an attempt to learn how to use it. 

You'll find a lot of scripts and notebooks in exercising some of the `ttnn` functionality. 

Some projects are good, some are bad. Don't laugh, and don't be mad. I'm learning as I go. :) 

## GPT-2

As I've learned `ttnn` through various notebooks, I found myself accidentally building a GPT-2 model from scratch with `torch`, and `ttnn`. 

Training was done in `torch`. Initially a small 124M model just using a CPU with 5K tokens or so. Then scaling up to building a medium 355M model from scratch and pretraining it on a 3 billion token dataset, `fineweb-3b`. 
