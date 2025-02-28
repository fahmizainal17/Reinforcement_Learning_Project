import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
import time
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

class GridWorld:
    """A simple grid world environment with obstacles."""
    
    def __init__(self, height=4, width=4):
        # Grid dimensions
        self.height = height
        self.width = width
        
        # Define states
        self.n_states = self.height * self.width
        
        # Actions: 0: up, 1: right, 2: down, 3: left
        self.n_actions = 4
        self.action_names = ['Up', 'Right', 'Down', 'Left']
        
        # Define rewards
        self.rewards = np.zeros((self.height, self.width))
        # Goal state
        self.rewards[self.height-1, self.width-1] = 1.0
        # Obstacles (negative reward)
        self.obstacles = []
        if height >= 4 and width >= 4:
            self.rewards[1, 1] = -1.0
            self.rewards[1, 2] = -1.0
            self.rewards[2, 1] = -1.0
            self.obstacles = [(1, 1), (1, 2), (2, 1)]
        
        # Start state
        self.start_state = (0, 0)
        
        # Goal state
        self.goal_state = (self.height-1, self.width-1)
        
        # Reset the environment
        self.reset()
    
    def reset(self):
        """Reset the agent to the start state."""
        self.agent_position = self.start_state
        return self._get_state()
    
    def _get_state(self):
        """Convert the agent's (row, col) position to a state number."""
        row, col = self.agent_position
        return row * self.width + col
    
    def _get_pos_from_state(self, state):
        """Convert a state number to (row, col) position."""
        row = state // self.width
        col = state % self.width
        return (row, col)
    
    def step(self, action):
        """Take an action and return next_state, reward, done."""
        row, col = self.agent_position
        
        # Apply the action
        if action == 0:  # up
            row = max(0, row - 1)
        elif action == 1:  # right
            col = min(self.width - 1, col + 1)
        elif action == 2:  # down
            row = min(self.height - 1, row + 1)
        elif action == 3:  # left
            col = max(0, col - 1)
        
        # Update agent position
        self.agent_position = (row, col)
        
        # Get reward
        reward = self.rewards[row, col]
        
        # Check if episode is done
        done = (row, col) == self.goal_state
        
        return self._get_state(), reward, done

class QLearningAgent:
    """A simple Q-learning agent."""
    
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        """Initialize the Q-learning agent."""
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        
        # Initialize Q-table
        self.q_table = np.zeros((n_states, n_actions))
        
        # Track visited states for visualization
        self.visit_counts = np.zeros(n_states)
        
        # Training metrics
        self.rewards_history = []
        self.exploration_rates = []
    
    def select_action(self, state):
        """Select an action using epsilon-greedy policy."""
        if np.random.random() < self.exploration_rate:
            # Explore: select a random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: select the action with the highest Q-value
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """Update the Q-table using the Q-learning update rule."""
        # Calculate the Q-target
        if done:
            q_target = reward
        else:
            q_target = reward + self.discount_factor * np.max(self.q_table[next_state])
        
        # Update the Q-value
        self.q_table[state, action] += self.learning_rate * (q_target - self.q_table[state, action])
        
        # Update visit count for visualization
        self.visit_counts[state] += 1
    
    def decay_exploration(self):
        """Decay the exploration rate."""
        self.exploration_rate *= self.exploration_decay
        self.exploration_rates.append(self.exploration_rate)
    
    def get_policy(self):
        """Return the current greedy policy."""
        return np.argmax(self.q_table, axis=1)
    
    def reset(self):
        """Reset the agent for a new training session."""
        self.q_table = np.zeros((self.n_states, self.n_actions))
        self.visit_counts = np.zeros(self.n_states)
        self.rewards_history = []
        self.exploration_rates = []


def create_gridworld_figure(env, agent, episode_count=0, total_reward=0):
    """Create a figure with environment, visit heatmap, and Q-values."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Episode: {episode_count}, Total Reward: {total_reward:.2f}, Exploration Rate: {agent.exploration_rate:.2f}")
    
    # Define colors for different cell types
    colors = {
        'empty': 'white',
        'obstacle': 'black',
        'goal': 'green',
        'start': 'blue',
        'agent': 'red'
    }
    
    # Helper function to draw grid
    def draw_grid(ax):
        # Create a grid
        for i in range(env.height + 1):
            ax.axhline(i, color='black', lw=1)
        for j in range(env.width + 1):
            ax.axvline(j, color='black', lw=1)
        
        # Set limits and remove ticks
        ax.set_xlim(0, env.width)
        ax.set_ylim(0, env.height)
        ax.invert_yaxis()  # Invert y-axis to match grid coordinates
        ax.set_xticks(np.arange(0.5, env.width, 1))
        ax.set_yticks(np.arange(0.5, env.height, 1))
        ax.set_xticklabels(range(env.width))
        ax.set_yticklabels(range(env.height))
    
    # Helper function to draw a cell
    def draw_cell(ax, row, col, cell_type):
        color = colors.get(cell_type, 'white')
        rect = patches.Rectangle((col, row), 1, 1, linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
        ax.add_patch(rect)
    
    # Helper function to draw an arrow
    def draw_arrow(ax, row, col, action):
        # Coordinates for arrows
        arrow_starts = {
            0: (col + 0.5, row + 0.7),  # up
            1: (col + 0.3, row + 0.5),  # right
            2: (col + 0.5, row + 0.3),  # down
            3: (col + 0.7, row + 0.5)   # left
        }
        
        arrow_ends = {
            0: (col + 0.5, row + 0.3),  # up
            1: (col + 0.7, row + 0.5),  # right
            2: (col + 0.5, row + 0.7),  # down
            3: (col + 0.3, row + 0.5)   # left
        }
        
        ax.annotate('', xy=arrow_ends[action], xytext=arrow_starts[action],
                  arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Draw Environment
    ax = axes[0]
    ax.set_title('GridWorld Environment')
    draw_grid(ax)
    
    # Draw cells
    for i in range(env.height):
        for j in range(env.width):
            if (i, j) in env.obstacles:
                draw_cell(ax, i, j, 'obstacle')
            elif (i, j) == env.goal_state:
                draw_cell(ax, i, j, 'goal')
            elif (i, j) == env.start_state:
                draw_cell(ax, i, j, 'start')
    
    # Draw agent
    row, col = env.agent_position
    draw_cell(ax, row, col, 'agent')
    
    # Draw policy arrows
    policy = agent.get_policy()
    for state in range(env.n_states):
        row, col = env._get_pos_from_state(state)
        if (row, col) not in env.obstacles and (row, col) != env.goal_state:
            draw_arrow(ax, row, col, policy[state])
    
    # Ensure proper aspect ratio
    ax.set_aspect('equal')
    
    # Draw Visit Heatmap
    ax = axes[1]
    ax.set_title('State Visitation Heatmap')
    draw_grid(ax)
    
    # Create heatmap data
    heatmap_data = np.zeros((env.height, env.width))
    for state in range(env.n_states):
        row, col = env._get_pos_from_state(state)
        heatmap_data[row, col] = agent.visit_counts[state]
    
    # Normalize values for coloring
    max_visits = max(1, np.max(heatmap_data))
    
    # Draw heatmap
    for i in range(env.height):
        for j in range(env.width):
            if (i, j) in env.obstacles:
                draw_cell(ax, i, j, 'obstacle')
            elif (i, j) == env.goal_state:
                draw_cell(ax, i, j, 'goal')
            else:
                intensity = heatmap_data[i, j] / max_visits
                color = plt.cm.viridis(intensity)
                rect = patches.Rectangle((j, i), 1, 1, linewidth=1, edgecolor='black', facecolor=color, alpha=0.7)
                ax.add_patch(rect)
                # Add visit count text
                if heatmap_data[i, j] > 0:
                    ax.text(j + 0.5, i + 0.5, int(heatmap_data[i, j]), ha='center', va='center', color='white' if intensity > 0.5 else 'black')
    
    # Ensure proper aspect ratio
    ax.set_aspect('equal')
    
    # Draw Q-values
    ax = axes[2]
    ax.set_title('Q-Values')
    draw_grid(ax)
    
    # Draw Q-values for each cell
    for state in range(env.n_states):
        row, col = env._get_pos_from_state(state)
        
        if (row, col) in env.obstacles:
            draw_cell(ax, row, col, 'obstacle')
            continue
        
        if (row, col) == env.goal_state:
            draw_cell(ax, row, col, 'goal')
            continue
        
        # Calculate q-values for each action
        q_values = agent.q_table[state]
        
        # Draw arrows proportional to Q-values
        for action in range(env.n_actions):
            q_value = q_values[action]
            
            # Only draw arrows for positive Q-values
            if q_value > 0:
                # Normalize arrow size
                max_q = max(0.1, np.max(q_values))
                arrow_size = 0.3 * (q_value / max_q)
                
                # Position calculations
                center_x = col + 0.5
                center_y = row + 0.5
                
                # Direction vectors
                directions = [
                    (0, -arrow_size),  # up
                    (arrow_size, 0),   # right
                    (0, arrow_size),   # down
                    (-arrow_size, 0)   # left
                ]
                
                dx, dy = directions[action]
                
                # Draw arrow
                ax.arrow(center_x, center_y, dx, dy, head_width=0.1, head_length=0.1, 
                       fc='blue', ec='blue', alpha=0.7)
                
                # Add Q-value text
                text_positions = [
                    (center_x, center_y - 0.25),  # up
                    (center_x + 0.25, center_y),  # right
                    (center_x, center_y + 0.25),  # down
                    (center_x - 0.25, center_y)   # left
                ]
                
                tx, ty = text_positions[action]
                ax.text(tx, ty, f"{q_value:.2f}", ha='center', va='center', fontsize=8, 
                       bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))
    
    # Ensure proper aspect ratio
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def create_metrics_figure(agent):
    """Create a figure with training metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot rewards
    if agent.rewards_history:
        axes[0].plot(agent.rewards_history)
        axes[0].set_title('Rewards per Episode')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].grid(True)
    else:
        axes[0].set_title('No reward data yet')
    
    # Plot exploration rate
    if agent.exploration_rates:
        axes[1].plot(agent.exploration_rates)
        axes[1].set_title('Exploration Rate Decay')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Exploration Rate (ε)')
        axes[1].grid(True)
    else:
        axes[1].set_title('No exploration rate data yet')
    
    plt.tight_layout()
    return fig

def train_single_episode(env, agent):
    """Train for a single episode and return the total reward."""
    state = env.reset()
    total_reward = 0
    done = False
    steps = 0
    max_steps = env.width * env.height * 3  # Prevent infinite loops
    
    while not done and steps < max_steps:
        # Select action
        action = agent.select_action(state)
        
        # Take the action
        next_state, reward, done = env.step(action)
        
        # Update the Q-table
        agent.update(state, action, reward, next_state, done)
        
        # Update state and total reward
        state = next_state
        total_reward += reward
        steps += 1
    
    # Decay exploration rate
    agent.decay_exploration()
    
    # Store the total reward
    agent.rewards_history.append(total_reward)
    
    return total_reward

def train_agent(env, agent, episodes, progress=gr.Progress()):
    """Train the agent for a specified number of episodes."""
    progress_text = ""
    progress(0, desc="Starting training...")
    
    for episode in progress.tqdm(range(episodes)):
        total_reward = train_single_episode(env, agent)
        
        if (episode + 1) % 10 == 0 or episode == episodes - 1:
            progress_text += f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Exploration: {agent.exploration_rate:.3f}\n"
    
    # Create final visualization
    env_fig = create_gridworld_figure(env, agent, episode_count=episodes, total_reward=total_reward)
    metrics_fig = create_metrics_figure(agent)
    
    return env_fig, metrics_fig, progress_text

def run_test_episode(env, agent):
    """Run a test episode using the learned policy."""
    state = env.reset()
    total_reward = 0
    done = False
    path = [env._get_pos_from_state(state)]
    steps = 0
    max_steps = env.width * env.height * 3  # Prevent infinite loops
    
    while not done and steps < max_steps:
        # Select the best action from the learned policy
        action = np.argmax(agent.q_table[state])
        
        # Take the action
        next_state, reward, done = env.step(action)
        
        # Update state and total reward
        state = next_state
        total_reward += reward
        path.append(env._get_pos_from_state(state))
        steps += 1
    
    # Create visualization
    env_fig = create_gridworld_figure(env, agent, episode_count="Test", total_reward=total_reward)
    
    # Format path for display
    path_text = "Path taken:\n"
    for i, pos in enumerate(path):
        path_text += f"Step {i}: {pos}\n"
    
    return env_fig, path_text, f"Test completed with total reward: {total_reward}"

def create_ui():
    """Create the Gradio interface."""
    # Create environment and agent
    env = GridWorld(height=4, width=4)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=1.0,
        exploration_decay=0.995
    )
    
    # Create initial visualizations
    init_env_fig = create_gridworld_figure(env, agent)
    init_metrics_fig = create_metrics_figure(agent)
    
    with gr.Blocks(title="Q-Learning GridWorld Simulator") as demo:
        gr.Markdown("# Q-Learning GridWorld Simulator")
        
        with gr.Tab("Environment Setup"):
            with gr.Row():
                with gr.Column():
                    grid_height = gr.Slider(minimum=3, maximum=8, value=4, step=1, label="Grid Height")
                    grid_width = gr.Slider(minimum=3, maximum=8, value=4, step=1, label="Grid Width")
                    setup_btn = gr.Button("Setup Environment")
                
                env_display = gr.Plot(value=init_env_fig, label="Environment")
            
            with gr.Row():
                setup_info = gr.Textbox(label="Environment Info", value="4x4 GridWorld with start at (0,0) and goal at (3,3)")
        
        with gr.Tab("Train Agent"):
            with gr.Row():
                with gr.Column():
                    learning_rate = gr.Slider(minimum=0.01, maximum=1.0, value=0.1, step=0.01, label="Learning Rate (α)")
                    discount_factor = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.01, label="Discount Factor (γ)")
                    exploration_rate = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.01, label="Initial Exploration Rate (ε)")
                    exploration_decay = gr.Slider(minimum=0.9, maximum=0.999, value=0.995, step=0.001, label="Exploration Decay Rate")
                    episodes = gr.Slider(minimum=10, maximum=500, value=100, step=10, label="Number of Episodes")
                    train_btn = gr.Button("Train Agent")
            
            with gr.Row():
                train_env_display = gr.Plot(label="Training Environment")
                train_metrics_display = gr.Plot(label="Training Metrics")
            
            train_log = gr.Textbox(label="Training Log", lines=10)
        
        with gr.Tab("Test Agent"):
            with gr.Row():
                test_btn = gr.Button("Test Trained Agent")
            
            with gr.Row():
                test_env_display = gr.Plot(label="Test Environment")
            
            with gr.Row():
                with gr.Column():
                    path_display = gr.Textbox(label="Path Taken", lines=10)
                    test_result = gr.Textbox(label="Test Result")
        
        # Setup environment callback
        def setup_environment(height, width):
            nonlocal env, agent
            env = GridWorld(height=int(height), width=int(width))
            agent = QLearningAgent(
                n_states=env.n_states,
                n_actions=env.n_actions,
                learning_rate=0.1,
                discount_factor=0.9,
                exploration_rate=1.0,
                exploration_decay=0.995
            )
            env_fig = create_gridworld_figure(env, agent)
            info_text = f"{height}x{width} GridWorld with start at (0,0) and goal at ({height-1},{width-1})"
            if env.obstacles:
                info_text += f"\nObstacles at: {env.obstacles}"
            return env_fig, info_text
        
        setup_btn.click(
            setup_environment,
            inputs=[grid_height, grid_width],
            outputs=[env_display, setup_info]
        )
        
        # Train agent callback
        def start_training(lr, df, er, ed, eps):
            nonlocal env, agent
            agent = QLearningAgent(
                n_states=env.n_states,
                n_actions=env.n_actions,
                learning_rate=float(lr),
                discount_factor=float(df),
                exploration_rate=float(er),
                exploration_decay=float(ed)
            )
            env_fig, metrics_fig, log = train_agent(env, agent, int(eps))
            return env_fig, metrics_fig, log
        
        train_btn.click(
            start_training,
            inputs=[learning_rate, discount_factor, exploration_rate, exploration_decay, episodes],
            outputs=[train_env_display, train_metrics_display, train_log]
        )
        
        # Test agent callback
        def test_trained_agent():
            nonlocal env, agent
            env_fig, path_text, result = run_test_episode(env, agent)
            return env_fig, path_text, result
        
        test_btn.click(
            test_trained_agent,
            inputs=[],
            outputs=[test_env_display, path_display, test_result]
        )
    
    return demo

if __name__ == "__main__":
    # Install required packages
    # !pip install gradio matplotlib numpy
    
    # Create and launch the UI
    demo = create_ui()
    demo.launch(share=True)
    