#!/bin/bash

SWAP_FILE=$1

sudo swapoff $SWAP_FILE
sudo fallocate -l 128G $SWAP_FILE
sudo mkswap $SWAP_FILE
sudo swapon $SWAP_FILE
