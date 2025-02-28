# **🤖 Q-Learning GridWorld Simulator**

[![HuggingFace](https://img.shields.io/badge/HuggingFace-Q--Learning%20Demo-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/fahmizainal17/Q-Learning_GridWorld_Simulator)

<div align="center">
    <a href="https://huggingface.co/spaces/fahmizainal17/Q-Learning_GridWorld_Simulator">
        <img src="https://img.shields.io/badge/Try%20The%20Demo-brightgreen?style=for-the-badge&logo=gradio" alt="Try Q-Learning Demo"/>
    </a>
</div>

---

## **📄 Overview**

The **Q-Learning GridWorld Simulator** by **Fahmi Zainal** is an interactive web application that demonstrates the fundamentals of reinforcement learning through a visual and intuitive interface. This project implements a Q-learning agent that learns to navigate through a grid environment with obstacles to reach a goal state. Users can modify learning parameters, observe the training process in real-time, and see how different settings affect the agent's learning capabilities. It's a perfect educational tool for understanding the core concepts of reinforcement learning.

---

## **Table of Contents**

1. [🎯 Objectives](#-objectives)
2. [🔧 Technologies Used](#-technologies-used)
3. [📝 Directory Structure](#-directory-structure)
4. [⚙️ Environment Setup](#️-environment-setup)
5. [🧠 Q-Learning Algorithm](#-q-learning-algorithm)
6. [🔍 Features](#-features)
7. [🖥️ Interface Components](#️-interface-components)
8. [💡 Parameter Optimization](#-parameter-optimization)
9. [📊 Visualization Components](#-visualization-components)
10. [🔄 Project Workflow](#-project-workflow)
11. [🚀 Running the Application](#-running-the-application)
12. [🌐 Deployment Options](#-deployment-options)
13. [🔮 Future Enhancements](#-future-enhancements)
14. [🎉 Conclusion](#-conclusion)
15. [📚 References](#-references)
16. [📜 License](#-license)

---

## **🎯 Objectives**

- **🎓 Educational Tool**: Provide an accessible way to understand reinforcement learning concepts
- **🧪 Experimentation Platform**: Allow users to observe how different parameters affect learning
- **👁️ Visualization**: Create intuitive visualizations of the Q-learning process
- **🔬 Interactive Learning**: Enable users to interact with and modify the learning environment
- **📱 Accessibility**: Make reinforcement learning concepts accessible through a web interface

---

## **🔧 Technologies Used**

![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-Interface-orange?style=for-the-badge&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAOCAMAAAAolt3jAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAaVBMVEUAAAD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQD/bQAAAADLj0KaAAAAInRSTlMABmip4vGkBQRltP7ysWMDA27n/uRoAQFx8/3vbgICaMnu58oS3AAAAAFiS0dEAIgFHUgAAAAJcEhZcwAACxMAAAsTAQCanBgAAAAHdElNRQfmAhsRLRrVeOltAAAAb0lEQVQI102NVw7DMAxDRdLDe3dnadz7X7JuA/QDBCgQBCSKQSgbUzaRaIZIlpW1ApUayErZ2tG4ONC6OtlzZXP3EMbDCZ5hVETpzUt4w5tFZe/g+AS8wHOLXrG+sRnev3iSXYz+faj+QBQD0fwBUGMI1RDqVCUAAAAASUVORK5CYII=)
![NumPy](https://img.shields.io/badge/NumPy-scientific%20computing-green?style=for-the-badge&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-visualization-red?style=for-the-badge&logo=matplotlib)

This project leverages several key technologies:

- **Python**: Core programming language for the implementation
- **NumPy**: For efficient numerical operations and array manipulation
- **Matplotlib**: For creating visualization components and plots
- **Gradio**: For building the interactive web interface
- **HuggingFace Spaces**: For hosting the deployed application

---

## **📝 Directory Structure**

```plaintext
.
├── LICENSE                        # Fahmi Zainal Custom License information
├── README.md                      # Project documentation
├── rl_gradio.py                   # Main application file with Gradio interface
├── requirements.txt               # Project dependencies
├── .github                        # GitHub configuration
│   └── workflows
│       └── huggingface-space-sync.yml  # Automatic deployment workflow
├── examples                       # Example screenshots and animations
│   ├── training_visualization.gif # Training process animation
│   ├── interface_components.png   # UI component overview
│   └── parameter_effects.png      # Visual comparison of parameters
└── space.yml                      # HuggingFace Spaces configuration
```

---

## **⚙️ Environment Setup**

### Local Development Environment

1. **Access the project via HuggingFace**:
   ```bash
   # Note: This is a proprietary project by Fahmi Zainal
   # Please contact the owner for access to the repository
   # Visit: https://huggingface.co/spaces/fahmizainal17/Q-Learning_GridWorld_Simulator
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python rl_gradio.py
   ```

### Deployment Environment

The application is configured for easy deployment to HuggingFace Spaces:

1. **Requirements file includes**:
   ```
   gradio>=4.0.0
   matplotlib>=3.5.0
   numpy>=1.20.0
   ```

2. **Space configuration (space.yml)**:
   ```yaml
   title: Q-Learning GridWorld Simulator
   emoji: 🤖
   colorFrom: blue
   colorTo: green
   sdk: gradio
   sdk_version: 4.0.0
   app_file: rl_gradio.py
   pinned: false
   author: fahmizainal17
   ```

---

## **🧠 Q-Learning Algorithm**

The core of this project is the Q-Learning algorithm, a model-free reinforcement learning technique that learns the value of actions in states through trial and error.

### Key Components

1. **Q-Table**: A matrix that stores expected rewards for each state-action pair
2. **Exploration vs. Exploitation**: Balancing random actions vs. using current knowledge
3. **Reward Function**: Positive reward at goal, negative at obstacles
4. **Update Rule**: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]

### Parameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| Learning Rate (α) | How quickly new information overrides old information | 0.01 - 0.5 |
| Discount Factor (γ) | How much future rewards are valued | 0.8 - 0.99 |
| Exploration Rate (ε) | Probability of taking a random action | 0.1 - 1.0 |
| Exploration Decay | Rate at which exploration decreases | 0.9 - 0.999 |

### Algorithm Pseudocode

```
Initialize Q-table with zeros
For each episode:
    Reset environment to starting state
    While not terminal state:
        With probability ε, select random action
        Otherwise, select action with highest Q-value
        Take action, observe reward and next state
        Update Q-table using: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        Move to next state
    Reduce exploration rate by decay factor
```

---

## **🔍 Features**

### 1. **Interactive Environment Configuration**
- Adjustable grid size (3×3 up to 8×8)
- Customizable obstacle placement
- Visual representation of the grid world

### 2. **Dynamic Parameter Adjustment**
- Real-time modification of learning parameters
- Immediate feedback on parameter changes
- Preset configurations for quick experimentation

### 3. **Live Training Visualization**
- Real-time updates of the Q-table during training
- Visual representation of the agent's policy
- Heatmap of state visitation frequency

### 4. **Performance Metrics**
- Reward history tracking
- Exploration rate visualization
- Episode completion statistics

### 5. **Testing and Evaluation**
- Test mode to evaluate learned policies
- Path visualization and analysis
- Performance comparison tools

### 6. **Educational Components**
- Interactive explanations of reinforcement learning concepts
- Step-by-step visualization of the learning process
- Comparative analysis of different parameter settings

---

## **🖥️ Interface Components**

The Gradio interface is divided into three main tabs:

### 1. **Environment Setup Tab**
- Grid size selection controls
- Environment visualization
- Environment information display

### 2. **Train Agent Tab**
- Learning parameter sliders
   - Learning Rate (α)
   - Discount Factor (γ)
   - Exploration Rate (ε)
   - Exploration Decay
- Episode count selection
- Training button
- Training visualizations
   - Environment state
   - Visit heatmap
   - Q-value visualization
- Training metrics
   - Reward chart
   - Exploration rate chart
- Training log display

### 3. **Test Agent Tab**
- Test execution button
- Path visualization
- Performance metrics display
- Path analysis tools

---

## **💡 Parameter Optimization**

### Impact of Different Parameters

| Parameter | Low Value Effect | High Value Effect |
|-----------|------------------|-------------------|
| Learning Rate (α) | Slow, stable learning | Fast, potentially unstable learning |
| Discount Factor (γ) | Focus on immediate rewards | Value future rewards more |
| Exploration Rate (ε) | Limited exploration | Extensive exploration |
| Exploration Decay | Quick transition to exploitation | Extended exploration phase |

### Recommended Configurations

1. **Balanced Learning** (Default):
   - Learning Rate: 0.1
   - Discount Factor: 0.9
   - Exploration Rate: 1.0
   - Exploration Decay: 0.995

2. **Fast Learning**:
   - Learning Rate: 0.3
   - Discount Factor: 0.8
   - Exploration Rate: 1.0
   - Exploration Decay: 0.95

3. **Thorough Exploration**:
   - Learning Rate: 0.05
   - Discount Factor: 0.95
   - Exploration Rate: 1.0
   - Exploration Decay: 0.998

---

## **📊 Visualization Components**

### 1. **GridWorld Environment**
- Shows the current state of the environment
- Highlights agent position, obstacles, and goal
- Displays learned policy with directional arrows

### 2. **State Visitation Heatmap**
- Color-coded visualization of state visit frequency
- Helps identify exploration patterns
- Reveals the agent's learned paths

### 3. **Q-Value Visualization**
- Displays learned Q-values as arrows with varying sizes
- Shows the relative value of different actions in each state
- Provides insight into the agent's decision-making process

### 4. **Training Metrics Charts**
- Reward per episode trend line
- Exploration rate decay visualization
- Convergence analysis tools

---

## **🔄 Project Workflow**

### Development Process

1. **Environment Design**:
   - Implement GridWorld class with customizable parameters
   - Define state transitions and reward structure
   - Create visualization utilities

2. **Agent Implementation**:
   - Develop Q-learning algorithm
   - Implement exploration strategies
   - Build tracking mechanisms for training metrics

3. **UI Development**:
   - Design the Gradio interface layout
   - Implement interactive components
   - Create dynamic visualizations

4. **Integration and Testing**:
   - Connect the backend reinforcement learning components with the UI
   - Test with various parameter configurations
   - Optimize performance and usability

5. **Deployment**:
   - Package the application for deployment
   - Configure HuggingFace Spaces integration
   - Set up GitHub Actions for automated updates

---

## **🚀 Running the Application**

### Local Execution

```bash
# Install requirements
pip install -r requirements.txt

# Run the application
python rl_gradio.py
```

This will start the Gradio server locally, typically accessible at `http://127.0.0.1:7860`.

### Using the Application

1. **Environment Setup**:
   - Set your desired grid size
   - Click "Setup Environment" to initialize

2. **Training**:
   - Adjust learning parameters as needed
   - Set the number of episodes
   - Click "Train Agent" to begin training
   - Observe the visualizations as training progresses

3. **Testing**:
   - After training, switch to the "Test Agent" tab
   - Click "Test Trained Agent" to see how it performs
   - Analyze the path taken and performance metrics

---

## **🌐 Deployment Options**

### HuggingFace Spaces

The application is configured for easy deployment to HuggingFace Spaces:

1. **Create a HuggingFace account** at https://huggingface.co/join

2. **Install the HuggingFace CLI**:
   ```bash
   pip install huggingface_hub
   ```

3. **Contact the project owner for deployment instructions**:
   ```bash
   # This project is owned by Fahmi Zainal
   # Please contact the owner for proper deployment instructions
   # The project is already deployed at:
   # https://huggingface.co/spaces/fahmizainal17/Q-Learning_GridWorld_Simulator
   ```

4. **For authorized collaborators only**:
   - Request proper access credentials from Fahmi Zainal
   - Follow the proprietary deployment guidelines provided by the owner

### Other Deployment Options

The application can also be deployed to:

- **Streamlit Cloud**: With minor modifications to use Streamlit instead of Gradio
- **Heroku**: Using a Procfile to specify the web process
- **Docker**: By containerizing the application for consistent deployment

---

## **🔮 Future Enhancements**

### Planned Features

1. **Additional Algorithms**:
   - SARSA implementation
   - Deep Q-Network (DQN) integration
   - Policy Gradient methods

2. **Enhanced Environments**:
   - Continuous state spaces
   - Stochastic environments
   - Multi-agent scenarios

3. **Advanced Visualizations**:
   - 3D environment representation
   - Animation of learning progress over time
   - Interactive policy exploration

4. **Educational Enhancements**:
   - Step-by-step algorithm explanations
   - Interactive tutorials
   - Challenge scenarios with specific learning objectives

5. **Performance Optimizations**:
   - Faster training algorithms
   - Parallel processing options
   - Pre-computed examples for instant demonstration

---

## **🎉 Conclusion**

The **Q-Learning GridWorld Simulator** developed by **Fahmi Zainal** provides an accessible and interactive platform for exploring reinforcement learning concepts. By visualizing the Q-learning process and allowing real-time parameter adjustments, it bridges the gap between theoretical understanding and practical implementation of reinforcement learning algorithms.

The project demonstrates how agents can learn optimal policies through trial and error, showcasing the power of Q-learning in a simple yet instructive environment. As an educational tool, it offers intuitive insights into the mechanics of reinforcement learning, making complex concepts more approachable for students, researchers, and AI enthusiasts.

This project represents Fahmi Zainal's work in the field of reinforcement learning visualization and is protected under a custom license that prohibits unauthorized use or distribution.

---

## **📚 References**

- Sutton, R. S., & Barto, A. G. (2018). [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html). MIT Press.
- OpenAI. (2018). [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/).
- Mnih, V., et al. (2015). [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236). Nature, 518(7540), 529-533.
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)
- [Gradio Documentation](https://gradio.app/docs/)

---

## **📜 License**

**Fahmi Zainal Custom License**

Copyright (c) 2025 Fahmi Zainal

Unauthorized copying, distribution, or modification of this project is prohibited. This project and its source code are the intellectual property of Fahmi Zainal. This is not free to copy or distribute. For inquiries about usage, licensing, or collaboration, contact the project owner.

All rights reserved.