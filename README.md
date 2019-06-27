# DDPG_Drone
This is a my final year project to controla  tiltrotor UAV through its transition between horizontal and forward flight. 
The Simulink files are models in which the agent block sends control inputs to the flight dynamics model. The FDM outputs generate a reward and observations which are used to update the Agent neural networks.
The agent uses deep deterministic policy gradients which is an off policy actor cirtic algorithm which updates towards the gradient of the greatest reward. A useful resource for this is: T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver and D. Wierstra, “Continuous control with deep reinforcement learning,” CoRR, vol. abs/1509.02971, 2016.
