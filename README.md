# CartPole Balancing with Deep Q-Networks (DQN)

This repository demonstrates how to train a **Deep Q-Network (DQN)** agent to master the classic **CartPole-v1** environment using **Stable-Baselines3** and **Gymnasium**. It includes step-by-step training rounds, custom performance tracking, and automated video recording of the agent's progress.

## 🚀 Features
* **Random Baseline:** Test a dummy random-action agent before training begins.
* **Iterative Training:** 20 rounds of structured training (5,000 timesteps per round) totaling 100,000 steps.
* **Evaluation & Visualization:** Automatically captures and writes evaluation metrics and gameplay videos (`.mp4`) across random, training, and testing phases.

## 🛠️ Installation & Setup

Ensure you have Python installed, then set up the required dependencies.

```bash
pip install gymnasium==0.28.1
pip install stable-baselines3[extra]
pip install matplotlib opencv-python
```

## 📦 Project Structure

* `1-random.mp4` — Gameplay footage of the untrained random agent.
* `2-training.mp4` — Visual evolution of the agent across different training phases.
* `3-testing.mp4` — Final evaluation performance video of the fully trained agent.
* `cartpole.dqn_X` — Saved model checkpoints for each training round.

## 💻 Usage

Run the primary script to initialize the environment, observe random behaviour, execute the training loops, and evaluate the final model checkpoint (`cartpole.dqn_19`):

```python
# Create environment
env = gym.make("CartPole-v1", render_mode="rgb_array")

# Define, train, and save model using Stable-Baselines3
model = sb3.DQN('MlpPolicy', env, learning_rate=0.0001, ...)
```

## 📊 Results

The agent continuously tracks evaluation logs displaying the average episode length and reward milestones over 20 structured checkpoints, eventually achieving the maximum stable balance threshold.

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
