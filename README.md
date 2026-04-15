# Audio-Based Species Classification with Classical and Quantum Feature Learning

## Project Overview  

This project develops a multi-label classification framework for identifying bird species from environmental audio recordings.

We consider a dataset:

$$
\{(x_i, y_i)\}_{i=1}^N, \quad x_i \in \mathbb{R}^{T}, \quad y_i \in \{0,1\}^K
$$

The objective is to learn:

$$
f_\theta: \mathbb{R}^{T} \rightarrow [0,1]^K
$$

such that:

$$
f_\theta(x)_k \approx \mathbb{P}(y_k = 1 \mid x)
$$

---

## Classical Learning Framework  

### Feature Representation  

Audio signals are transformed into time-frequency space:

$$
\phi: \mathbb{R}^{T} \rightarrow \mathbb{R}^{F \times \tau}
$$

where $\phi(x)$ is a log-mel spectrogram.

---

### Model  

$$
f_\theta(x) = \sigma\big(g_\theta(\phi(x))\big)
$$

We optimize:

$$
\mathcal{L}(\theta) = - \sum_{i=1}^N \sum_{k=1}^K w_k \left[ y_{ik} \log \hat{y}_{ik} + (1 - y_{ik}) \log(1 - \hat{y}_{ik}) \right]
$$

---

## Quantum Feature Learning (Planned Work)

### Research Direction  

Over the next several months, this project will extend the classical pipeline into a **quantum-enhanced representation framework**, grounded in two complementary approaches:

1. **Quantum Vision (QV) wave representations**  
2. **Quantum Time-Series Encoding (QTSE) with Quantum Audio Neural Networks (QANN)**  

The central objective is to move beyond fixed spectrogram representations and instead construct **wave-based and entangled quantum representations of audio signals**.

---

## Quantum Vision (QV) Representation  

Classical spectrograms $I(x,y)$ are treated as collapsed observations of a richer underlying structure. Following Quantum Vision theory, we construct **information wave functions** from spectrograms.

### Basis Wave Functions  

For spatial shifts $m \in \{\pm1, \pm2\}$:

$$
\psi_{x,m}(x,y) = I(x - m, y) - I(x,y)
$$

$$
\psi_{y,m}(x,y) = I(x, y - m) - I(x,y)
$$

These define **8 basis wave functions** capturing local spectral variation.

---

### Wave Superposition  

A general wave function is constructed via superposition:

$$
\psi = \sum_{m=-2}^{2} a_m \psi_{x,m}(x,y) + b_m \psi_{y,m}(x,y)
$$

Nonlinear combinations are learned via convolution:

$$
\psi = \sum_{m=-2}^{2} \left(
\text{ReLU}(H_m * \psi_{x,m}) + \text{ReLU}(V_m * \psi_{y,m})
\right)
$$

This produces **wave-based feature maps**:

$$
\psi_{128} \in \mathbb{R}^{128 \times F \times \tau}
$$

These representations emphasize **boundaries, transitions, and spectral dynamics**, rather than static intensities. :contentReference[oaicite:2]{index=2}  

---

## Quantum Time-Series Encoding (QTSE)

To capture temporal structure explicitly, we adopt a quantum encoding scheme for audio.

### Quantum State Representation  

Audio is encoded as:

$$
|A\rangle = \frac{1}{\sqrt{2^n}} \sum_{T=0}^{2^n-1} |f(T)\rangle |T\rangle
$$

where:
- $|f(T)\rangle$: timbre (e.g., MFCC features)  
- $|T\rangle$: temporal index  

This creates a **joint representation of content and time**. :contentReference[oaicite:3]{index=3}  

---

### Entangled Encoding  

QTSE uses **two entangled qubit registers**:

- Timbre register:  
$$
f(T) = A_0^T A_1^T \dots A_{q-1}^T
$$

- Time register:  
$$
|T\rangle = |t_0 t_1 \dots t_{n-1}\rangle
$$

Combined state:

$$
|A\rangle = \frac{1}{\sqrt{2^n}} \sum_{T} \left( \bigotimes_{i=0}^{q-1} |A_i^T\rangle \right) \otimes |T\rangle
$$

This encoding preserves:
- Spectral structure (timbre)  
- Temporal dependencies (sequence order)  

---

## Quantum Audio Neural Network (QANN)

The encoded quantum state is processed via a parameterized quantum circuit:

$$
|\psi_{\text{out}}\rangle = U(\theta) |A\rangle
$$

where:
- $U(\theta)$ is a unitary transformation  
- constructed from rotation and entanglement gates  

Measurement yields:

$$
f(x) = \langle \psi_{\text{out}} | O | \psi_{\text{out}} \rangle
$$

This defines a hybrid model:

$$
f(x) = h_\theta\big(\langle \psi | O | \psi \rangle\big)
$$

---

## Fourier-Based Encoding Connection  

Audio preprocessing already involves:

$$
\hat{x}(\omega) = \int x(t) e^{-i \omega t} dt
$$

This aligns naturally with quantum encoding because:
- Spectrograms represent **frequency-domain structure**
- Quantum phase rotations can encode frequency information:

$$
R_Z(\theta_j), \quad \theta_j \propto \hat{x}(\omega_j)
$$

Thus, Fourier features serve as a **bridge between classical signal processing and quantum state preparation**.

---

## Planned Timeline (Next Few Months)

The quantum extension will be developed in stages:

### Phase 1: QV Feature Integration  
- Implement wave-based transformations on spectrograms  
- Compare CNN vs QV-CNN representations  

### Phase 2: Quantum Encoding (QTSE)  
- Encode MFCC + temporal indices into quantum states  
- Simulate encoding using classical backends  

### Phase 3: Hybrid Quantum Model  
- Implement parameterized quantum circuits  
- Integrate with classical classifier (hybrid QANN)  

### Phase 4: Evaluation  
- Compare against classical baselines  
- Analyze robustness in noisy soundscapes  

---

## Research Contribution  

This project proposes a unified framework combining:

- Classical deep learning for feature extraction  
- Quantum wave representations (QV)  
- Entangled temporal encoding (QTSE)  

The core hypothesis is:

> Audio classification performance can be improved by moving from fixed spectrogram representations to **wave-based and entangled quantum representations** that preserve richer structural information. Specfically in the application of species recognition using Perimeter monitoring of forest areas.

---

## Long-Term Vision  

- Scalable quantum audio classifiers  
- Improved generalization under noise  
- Efficient representation learning with fewer parameters  
- Deployment on near-term quantum hardware (NISQ systems)  
