# AlphaGo Zero Recreation: Implementing and Optimizing for 5x5 Go

## Project Overview

This project aimed to recreate DeepMind's AlphaGo Zero algorithm, adapting it for the smaller 5x5 Go board. The goal was to gain deep insights into state-of-the-art reinforcement learning systems, explore implementation challenges, and lay groundwork for future game AI innovations.

## Technical Implementation

### 1. Neural Network Architecture

- Custom feedforward network using PyTorch
- 4 fully-connected layers, width of 200 neurons, ReLU activation
- Key innovations:
  a) Initialization based on "Exponential expressivity in deep neural networks through transient chaos"
  b) Zero-initialization of final linear layer weights

### 2. Input Representation

- Enhanced board state representation
- Input includes:
  - Last 8 board states
  - Two planes indicating current player

### 3. Monte Carlo Tree Search (MCTS)

- Efficient tree traversal and expansion
- Correct reward propagation
- Exploration-exploitation balance using UCB1

### 4. Self-Play and Training

- Dirichlet noise for move diversity
- On-the-fly training example generation
- Data augmentation through rotations and flips

### 5. Evaluation Metrics

- Win rate tracking
- Elo rating system
- MCTS rollouts and move probabilities visualization

### 6. Development Environment

- Language: Python
- ML Framework: PyTorch
- Visualization: Pygame
- Version Control: GitHub
- Performance Tracking: TensorBoard

## Optimization and Challenges

1. **Sample Efficiency**: Initial dataset from random playouts
2. **Training Stability**: Data augmentation breakthrough
3. **Computational Efficiency**: GPU acceleration, batching operations
4. **Debugging and Validation**: Leveraged personal Go expertise (~4kyu level)

## Results and Analysis

- Hardware: Razer Blade 15 laptop, RTX 2070 GPU
- Training Duration: ~3.5 days to performance plateau
- Achievements:
  - 2000+ Elo point gain
  - Near-perfect play on 5x5 board
  - Demonstration of advanced Go concepts

## Future Work

1. **Uncertainty Quantification**
   - Bayesian neural networks
   - Evidential deep learning
   - MCTS enhancement with uncertainty

2. **Continual Learning**
   - Elastic weight consolidation
   - Progressive neural networks
   - Dynamic replay buffer management

3. **Scaling to Larger Boards**
   - Architecture adaptation
   - MCTS optimization
   - Knowledge transfer techniques

4. **Sample Efficiency Improvements**
   - Prioritized experience replay
   - Curriculum learning

5. **Interpretability and Visualization**
   - Feature visualization
   - Decision process explanation tools

6. **Human Benchmarking**

7. **Code Refactoring**
   - Exploration of functional programming paradigms

## Conclusion

This AlphaGo Zero recreation demonstrates the implementation of a complex AI system, showcasing problem-solving skills in MCTS implementation, neural network optimization, and performance debugging. The project's modular design and GitHub documentation provide a foundation for future game AI and reinforcement learning research.

The combination of theoretical knowledge and practical skills highlights the depth of understanding gained. While focused on 5x5 Go, the insights have potential applications in larger-scale AI problems and other domains.

This project serves as a stepping stone towards advanced research in uncertainty quantification and continual learning in reinforcement learning systems. It demonstrates the ability to independently implement and optimize cutting-edge AI algorithms, providing a strong foundation for future contributions to artificial intelligence.
