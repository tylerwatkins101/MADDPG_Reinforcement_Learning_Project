## Project Details

In this project I use the MADDPG algorithm to train a Reinforcement Learning Agent to improve it's performance in the Tennis environment provided by Unity.

![Tennis](photos/tennis.png)

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). 

Specifically, after each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores. This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5

## Software Dependencies

The project was done using python 3 and has the following library dependencies that can be installed using pip:

- unityagents
- numpy
- torch
- matplotlib

## File Descriptions

- Tennis.ipynb - jupyter notebook used to open environment, instantiate and train agent, and save trained model parameters.
- model.py - python module containing Actor-Network and Critic-Network architecture classes.
- ddpg_agent.py - python module containing DDPG Agent, Replay Buffer, and OU Noise classes.
- checkpoint_actor.pth - saved trained actor model parameters.
- checkpoint_critic.pth - saved trained critic model parameters
- Report.md - explains the model parameters, training, and potential future improvements.

## Instructions

To run the code and train the model yourself you'll need to download the Unity environment as well Tennis.ipynb, model.py, and ddpg_agent.py from this repository.

Follow the instructions below to activate the unity environment on your machine.

(For Windows users) The ML-Agents toolkit supports Windows 10. While it might be possible to run the ML-Agents toolkit using other versions of Windows, it has not been tested on other versions. Furthermore, the ML-Agents toolkit has not been tested on a Windows VM such as Bootcamp or Parallels.

Download the Unity Environment
For this project, you will not need to install Unity - this is because we have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

Linux: click here
Mac OSX: click here
Windows (32-bit): click here
Windows (64-bit): click here
Then, place the file in the p3_collab-compet/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

(For AWS) If you'd like to train the agent on AWS (and have not enabled a virtual screen), then please use this link to obtain the "headless" version of the environment. You will not be able to watch the agent without enabling a virtual screen, but you will be able to train the agent. (To watch the agent, you should follow the instructions to enable a virtual screen, and then download the environment for the Linux operating system above.)

Open the Tennis.ipynb jupyter notebook in your browser and run the code cells from top to bottom. In the notebook you'll see a visualization of the agent learning the task and after the agent has scored an average of at least +0.5 over 100 consecutive episodes the agent's model parameters will be saved to a .pth file.

