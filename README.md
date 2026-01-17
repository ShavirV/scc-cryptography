# Classical and Quantum Cryptography

## Overview

This project provides a comprehensive comparison of **classical** and **quantum** approaches to cryptographic problems, specifically:

- **Classical Approach**: Recurrent Neural Networks (RNNs) for password generation and hash cracking
- **Quantum Approach**: Shor's Algorithm and QAOA (Quantum Approximate Optimization Algorithm) implementations

### Research Goals

1. **Performance Comparison**: Benchmark RNN training and inference on an AMD EPYC CPU and NVIDIA A100 GPU
2. **Quantum Simulation**: Compare Shor's Algorithm performance on CPU (AMD EPYC), GPU (A100), and IBM QPU
3. **Hybrid Approaches**: Explore QAOA-inspired training for classical neural networks
4. **Cryptographic Analysis**: Evaluate the effectiveness of different approaches to password/hash problems

### 1. Neural Network Implementations

#### **`basic-net/`** - Basic RNN
Simple character-level RNN for password generation.
- `model.py` - Basic LSTM architecture
- `train.py` - Training loop
- `generate.py` - Password generation
- `data_prep.py` - Data preprocessing

#### **`better-net/`** - Enhanced RNN
Improved RNN with better architecture and training strategies.
- Enhanced LSTM with dropout and regularization
- Temperature-based sampling
- Multiple generation strategies
- Pre-trained model included (`password_rnn.pth`)

#### **`hash-net/`** - Hash Cracker RNN
QAOA-inspired neural network for hash cracking.

**Key Features:**
- Multi-objective loss function (hash matching + character prediction + diversity)
- Differentiable hash approximation
- Smart generation with temperature sampling
- Repetition prevention mechanisms

**Files:**
- `hash_model.py` - Neural architecture
- `qaoa_train.py` - QAOA-inspired training
- `hash_generate.py` - Hash cracking generation
- `simple_hash.py` - Differentiable hash function
- `verify_model.py` - Model verification

<!--**Performance:**
- Hash Similarity: 40-60%
- Training Time: 30-60 minutes (150 epochs)
- Generation Speed: ~100ms per attempt-->

#### **`Neural net/`** - Original Implementation
Initial prototype and experimental code.

---

## Installation

### Prerequisites

```bash
# System packages (Rocky Linux / RHEL)
sudo dnf groupinstall -y "Development Tools"
sudo dnf install -y cmake git wget curl openssl-devel zlib-devel \
    openblas-devel lapack-devel openmpi openmpi-devel

# Enable MPI
export PATH=/usr/lib64/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib64/openmpi/lib:$LD_LIBRARY_PATH
```

### Python Environment Setup

#### For RNN Components (PyTorch)

```bash
# Create virtual environment
python3 -m venv pytorch_env
source pytorch_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio numpy matplotlib
```

#### For Quantum Components (Qiskit)

```bash
# Create separate virtual environment
python3 -m venv qiskit_env
source qiskit_env/bin/activate

# Install Qiskit
pip install --upgrade pip
pip install qiskit[all] qiskit-aer qiskit-ibm-runtime
```

### Clone Repository

```bash
git clone <repository-url>
cd classical-and-quantum-cryptography
```

---

## Usage

### Neural Network Training

#### Basic Password Generation RNN

```bash
cd better-net
source ~/pytorch_env/bin/activate

# Generate training data
python generate_training_data.py --count 10000 --output passwords.txt

# Train model
python train.py

# Generate passwords
python generate.py
```

#### Hash Cracker RNN

```bash
cd hash-net
source ~/pytorch_env/bin/activate

# 1. Generate training data
python generate_training_data.py --count 10000 --output passwords.txt

# 2. Train the hash cracker
python qaoa_train.py

# 3. Test hash cracking
python hash_generate.py

# 4. Verify model performance
python verify_model.py
```

**Advanced Usage:**

```python
from smart_generate import smart_crack_hash
from simple_hash import SimpleHashFunction

hash_function = SimpleHashFunction()
target_hash = hash_function.actual_hash("mypassword123")

cracked, similarity = smart_crack_hash(
    target_hash,
    max_attempts=20,
    max_length=20
)
```

---

### Quantum Algorithm Execution

#### Shor's Algorithm - Simulation

```bash
cd shorsTest
source ~/qiskit_env/bin/activate

# Run on simulator
python3 shor_simulation.py

# Generic implementation (any composite number)
python3 shor_generic.py
```

#### Shor's Algorithm - Real Quantum Hardware

```bash
# Configure IBM Quantum credentials
# Set up your IBM Quantum account at https://quantum-computing.ibm.com/

# Run on IBM QPU
python3 shor_actual.py
```

#### Batch Benchmarking

```bash
# Run comprehensive benchmark suite
./batch_run.sh

# Analyze results
python3 analyze.py

# View generated plots
ls plots/
```

---

### Password Generation & Hashing

#### Generate Custom Hash Dataset

```bash
# Generate 10,000 passwords with custom hash
./Custom_Password_Hash_Generator.sh 10000

# Output: custom_password_hashes.txt
# Format: password: hash
```

#### Generate Multi-Algorithm Hash Dataset

```bash
# Generate 10,000 passwords with multiple hash algorithms
./Actual_Password_Hash_Generator.sh 10000

# Output: actual_password_hashes.txt
# Format:
# password
#   SHA-1:     hash
#   SHA-3:     hash
#   SHA-256:   hash
#   SHA-512:   hash
#   SHA-512Q:  hash
```

#### Hash Single Password

```bash
# Custom hash algorithm
./Hash_Password.sh 'SecurePassword123!'

# Output: Password and 64-character hex hash
```

---

<!--## Results

### Expected Performance Characteristics

#### RNN Training (100 epochs, 10K passwords)

| Hardware | Training Time | Inference (1000 passwords) | Memory Usage |
|----------|---------------|---------------------------|--------------|
| AMD EPYC (64 cores) | ~45-60 min | ~5-10 sec | ~4 GB |
| NVIDIA A100 (40GB) | ~10-15 min | ~1-2 sec | ~8 GB |

#### Hash Cracker RNN Performance

| Metric | Value |
|--------|-------|
| Hash Similarity | 40-60% |
| Exact Match Rate | 0% (current implementation) |
| Training Time (150 epochs) | 30-60 minutes |
| Generation Speed | ~100ms per attempt |
| Model Parameters | ~1.5 million |-->

<!--------->
<!--
## Architecture Details

### RNN Architecture (Hash Cracker)

```
Input: Target Hash (128-bit binary vector)
  ↓
Hash Encoder (Linear 128 → 128)
  ↓
Character Embeddings (Vocab size: 72, Embedding dim: 64)
  ↓
2-Layer LSTM (Hidden size: 256, Dropout: 0.2)
  ↓
Output Layer (Linear 256 → 72)
  ↓
Character Probabilities (Softmax)
```

**Loss Function (QAOA-inspired):**
```
Total Loss = α × Hash_Loss + β × Char_Loss + γ × Diversity_Loss

where:
  α = 1.0   (hash matching weight)
  β = 0.1-0.2 (character prediction weight)
  γ = 0.05  (diversity penalty weight)-->
<!--```-->
<!--

## Research Applications

### Cryptographic Security Analysis
- **Password Strength Evaluation**: Test password generation strategies
- **Hash Function Robustness**: Evaluate resistance to ML-based attacks
- **Post-Quantum Cryptography**: Assess quantum algorithm threat to current systems

### Algorithm Benchmarking
- **Classical vs Quantum**: Direct performance comparison
- **Hardware Evaluation**: CPU vs GPU vs QPU characteristics
- **Scalability Analysis**: Performance as problem size increases

### Machine Learning Research
- **QAOA-Inspired Training**: Novel loss functions for optimization
- **Hybrid Classical-Quantum**: Combining strengths of both approaches
- **Adversarial Training**: Neural networks vs cryptographic primitives

---

## Troubleshooting

### Common Issues

#### "No training pairs created"
```bash
# Solution: Regenerate training data
python generate_training_data.py --count 10000 --output passwords.txt
```

#### "CUDA out of memory"
```python
# Solution: Reduce batch size
# Edit train.py:
batch_size = 32  # Instead of 64
```

#### "Device mismatch errors"
```python
# Solution: Ensure all tensors on same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
input_tensor = input_tensor.to(device)
```

#### Qiskit Authentication Error
```bash
# Solution: Configure IBM Quantum credentials
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")
```
