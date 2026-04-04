#!/bin/bash
set -e

# get_datasets.sh
# Downloads CIFAR-10 and MNIST to data/

# Navigate to project root (assuming script is in scripts/)
cd "$(dirname "$0")/.."

mkdir -p data/cifar-10
mkdir -p data/mnist

echo "====================================="
echo "Downloading CIFAR-10 binary version"
echo "====================================="
curl -L https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -o data/cifar-10-binary.tar.gz

echo "Extracting CIFAR-10..."
tar -xzf data/cifar-10-binary.tar.gz -C data/cifar-10 --strip-components=1
rm data/cifar-10-binary.tar.gz
echo "CIFAR-10 extracted to data/cifar-10."

echo ""
echo "====================================="
echo "Downloading MNIST"
echo "====================================="
MIRROR="https://ossci-datasets.s3.amazonaws.com/mnist"
FILES=("train-images-idx3-ubyte.gz" "train-labels-idx1-ubyte.gz" "t10k-images-idx3-ubyte.gz" "t10k-labels-idx1-ubyte.gz")

for f in "${FILES[@]}"; do
    echo "Fetching $f..."
    curl -L "$MIRROR/$f" -o "data/mnist/$f"
    echo "Extracting $f..."
    gunzip -f "data/mnist/$f"
done
echo "MNIST extracted to data/mnist."

echo ""
echo "Datasets successfully downloaded to the data/ directory!"
