# DLDNN
A Neural Network Model for DLD Design and Simulation
DLDNN is design automation platform for deterministic lateral displacement (DLD) device that incorporates the power of neural network and multi-objective genetic algorithm.
## Design Automation
The figure blow shows the design automation process in DLDNN platform. First a set of inputs consisting D_1, and D_2, which are the diameter of particles to be separated, desired condition to be applied on f, N, Re and finally the desired ratio between flexibility and stability of the design. By knowing the inputs the algorithm tune the parameters f, N, Re, and G in order to satisfy the optimization objective which was set in input.The importance of using flexibility and stability index comes from the nature of the adjustable parameters.f and N are two geometrical parameters that can not be changed after fabricating the design. On the other hand, Re is a fluid characteristic that can be altered by simply adjusting the fluid flow. So, these to terms are introduced here to predict the range of the proposed design to cover a variety of critical diameter and its immunity to change it due to some fluctuation and fabrication inaccuracy.
Afeter reciving inputs the multi-objective algorithm (NSGA3) Creates the population by mutatins and crossovers. Afterward the f, N, and Re goes through the pre-trained ANN to predict the critical diameter(D_c) the G which is the gap between pillars is used for mapping and re scaling porpuses cause all the fields are normalized for neural network training. By extracting critical diamter the cost function can be calculated and the convergence criteria can be checked.
Algorithm outputs the optimum f, N, Re, G, and bandwidth. bandwidth is the difference between tha maximum and minimum critical diameter of a particular device and it is used for an index if flexibilty and stability. In addition to the outputs by using pre-trained convolutional neural network DLDNN provides the ability for simulating particles trajectories. it should be noted that the number of periods for particles trajectories simulation must be specified.

For more information check: link will be added soon

![2-Design Automation](https://user-images.githubusercontent.com/97515569/179344205-92cddf73-6da5-44d6-9d3e-daed59f4f94e.png)

## Files Descriptions
###### utility functions 
- DLD_utils.py: Containing the function for extracting data from numerical simulation and mapping the generated field
- DLD_env.py: Containing the function for post processing the data from numerical simulation 
- particle_trajectory.py: simulate particle trajectory by having the velocity fields in horizontal and vertical direction

###### Concolutional Neural Network
- generate_data.py: Dataset generation for concolutional neural network
- Conv_base.py: The convolutional neural network class containng the network architecture
- Conv_net_train.py: The training process of Conculutional neural network
- Temp9: The file is not available in the repository it is accessable through this Link(https://drive.google.com/drive/folders/1--o_9SYRY1sq_FOjo_ogZG6z_9ejrwNq?usp=sharing)

###### Fully Connceted Neural Network 
- Direct_network_generate_data.py: Gets the labels to fileds dataset and change it into filed to critical diameter datset
- Direct_NN.py: The  fully conncted neural network class containing structure of the neural network, trainng process
- - DNN_model_hlayers8_nodes_128.h5: The trained Direct neural network wights and bais 
###### Design Automation 
- Design_Aut.py: Main code of the design automation platform






