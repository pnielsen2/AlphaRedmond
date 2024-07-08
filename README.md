# AlphaGo Zero Recreation: Implementing and Optimizing for 5x5 Go

## Project Overview

This project aimed to recreate DeepMind's AlphaGo Zero algorithm, adapting it for the smaller 5x5 Go board. The goal was to gain deep insights into the workings of state-of-the-art reinforcement learning systems, explore the challenges of implementing such algorithms from scratch, and lay the groundwork for future innovations in game AI and reinforcement learning.

## Technical Implementation

### Neural Network Architecture

I implemented a custom feedforward network using PyTorch, consisting of 4 fully-connected layers with a width of 200 neurons each and ReLU activation functions. Two key innovations in the network design were:

- **Initialization based on the paper "Exponential expressivity in deep neural networks through transient chaos,"** which optimized training dynamics from a chaos-theory perspective. This resulted in noticeable improvements in training speed.
- **Zero-initialization of the final linear layer weights,** ensuring initial value predictions of 0 and uniform policy distribution. This approach prevented bias from random initialization, allowing for more thorough exploration of the game tree.

### Input Representation

I enhanced the standard board state representation by adding an extra plane to indicate the current player. This modification addressed a subtle bias in the original implementation where the white player had a slight advantage in edge recognition. The input to the network included:

- Last 8 board states
- Two planes indicating the current player (all zeros/ones and its inverse)

### Monte Carlo Tree Search (MCTS)

Implementing MCTS proved to be one of the most challenging aspects of the project. Key considerations included:

- Efficient tree traversal and expansion
- Correct propagation of rewards, accounting for alternating players
- Balancing exploration and exploitation using the UCB1 formula

### Self-Play and Training

The system generated training data through self-play. Notable features include:

- Addition of Dirichlet noise to the root node of MCTS to ensure move diversity
- On-the-fly generation of training examples from stored games, improving memory efficiency by a factor of 8
- Data augmentation through random rotations and flips, effectively increasing the dataset size by 8x

### Evaluation Metrics

To track the AI's progress, I implemented several evaluation methods:

- Win rate tracking (especially important as 5x5 Go is solved for black)
- Elo rating system, playing against past versions of the model
- Visualization of MCTS rollouts and move probabilities

## Development Environment and Tools

- **Primary language:** Python
- **Machine learning framework:** PyTorch
- **Visualization:** Pygame for game state rendering
- **Version control and sharing:** GitHub
- **Performance tracking:** TensorBoard for logging and visualizing training progress

## Optimization and Challenges

### Sample Efficiency

To improve learning speed, I created an initial dataset from random playouts. This allowed for hyperparameter tuning and provided a baseline for evaluating improvements.

### Training Stability

Early iterations faced slow convergence. The breakthrough came with the implementation of data augmentation, which significantly accelerated initial learning.

### Computational Efficiency

Given the resource constraints of a personal project, optimizing for computational efficiency was crucial. This involved careful batching of operations and strategic use of GPU acceleration where available.

### Debugging and Validation

As an experienced Go player (~4kyu level), I leveraged my game knowledge to validate the AI's behavior and catch subtle errors. This domain expertise was crucial in diagnosing issues and ensuring the correctness of the MCTS implementation.

## Results and Analysis

The training process was conducted on a Razer Blade 15 gaming laptop with an RTX 2070 graphics card. A typical training run to reach performance plateau took approximately 3.5 days. After this training period, the model showed remarkable improvement:

- Gained over 2000 Elo points from its initial random play
- Achieved near-perfect play on the 5x5 board
- Demonstrated strategic concepts such as influence and territory control

Interestingly, the AI sometimes developed unconventional move sequences that, while initially surprising, proved to be highly effective. This highlights the potential for AI to discover novel strategies in even well-studied games.

## Reflections and Future Work

This project provided deep insights into the intricacies of implementing advanced AI algorithms. It highlighted the importance of seemingly small details, like input representation, in the overall performance of the system. Building on this foundation, several exciting directions for future research and development have emerged:

### Uncertainty Quantification

Incorporating uncertainty quantification into the model could significantly enhance its decision-making capabilities and robustness. Potential approaches include:

- Implementing Bayesian neural networks to provide probability distributions over move values and policy outputs
- Exploring evidential deep learning techniques to directly learn uncertainty estimates
- Investigating how uncertainty information could be leveraged in the MCTS algorithm to improve exploration and decision-making
- Studying the impact of uncertainty-aware models on sample efficiency and overall performance

### Continual Learning and Preventing Catastrophic Forgetting

A key area of interest is developing techniques to enable continual learning and mitigate catastrophic forgetting. This could involve:

- Implementing elastic weight consolidation or similar techniques to preserve important knowledge while learning new strategies
- Exploring progressive neural networks or dynamically expanding architectures to accommodate new knowledge without disrupting existing capabilities
- Developing meta-learning approaches to enable rapid adaptation to new game states or strategies
- Investigating how these techniques could allow for a smaller, more dynamic replay buffer that can discard outdated moves more quickly without losing critical information

### Scaling to Larger Board Sizes

Adapting the current implementation to larger Go boards (9x9, 19x19) would present new challenges and opportunities:

- Revisiting the neural network architecture, potentially reintroducing convolutional layers
- Optimizing the MCTS algorithm for deeper and wider search trees
- Investigating techniques to transfer knowledge from smaller to larger board sizes

### Improving Sample Efficiency

Building on the current work, future efforts could focus on further enhancing sample efficiency:

- Implementing prioritized experience replay
- Exploring curriculum learning strategies
- Investigating how continual learning techniques could reduce reliance on large replay buffers

### Interpretability and Visualization

Developing more advanced tools for interpreting the model's decision-making process could provide valuable insights:

- Visualizing learned features and their activation patterns
- Creating heat maps of move probabilities and value estimates across the board
- Developing tools to explain the AI's strategic reasoning in human-understandable terms

### Human Benchmarking

While the current implementation has not been formally tested against human players, conducting such benchmarks could provide valuable insights into the AI's true playing strength and decision-making processes.

### Code Refactoring and Optimization

The project was initially developed using object-oriented programming principles for modularity. Future iterations could explore functional programming paradigms, which may offer simplicity and flexibility for machine learning applications.

## Conclusion

This AlphaGo Zero recreation for 5x5 Go demonstrates not only the technical implementation of a complex AI system but also showcases problem-solving skills in dealing with challenges like efficient MCTS implementation, neural network optimization, and performance debugging. The project's modular design and thorough documentation on GitHub make it a solid foundation for future enhancements and research in game AI and reinforcement learning.

The combination of theoretical knowledge (as demonstrated by the neural network initialization technique) and practical implementation skills highlights the depth of understanding gained through this project. While focused on a smaller board size, the insights and techniques developed here have potential applications in larger-scale AI problems and other domains beyond Go.

This project serves as a stepping stone towards more advanced research in areas like uncertainty quantification and continual learning in reinforcement learning systems. It demonstrates the ability to independently implement and optimize cutting-edge AI algorithms, providing a strong foundation for future contributions to the field of artificial intelligence.

The success of this implementation on a smaller scale opens up exciting possibilities for applying similar techniques to more complex problems. The insights gained from this project, particularly in areas like efficient training and novel strategy discovery, could have far-reaching implications in fields such as decision-making under uncertainty, strategic planning, and adaptive learning systems.

Ultimately, this project not only achieved its goal of recreating a sophisticated AI algorithm but also laid the groundwork for future innovations in reinforcement learning and AI. It stands as a testament to the power of combining theoretical understanding with practical implementation, and serves as a launching pad for exploring the frontiers of artificial intelligence.
