# DLDNN
DLDNN is design automation platform for deterministic lateral displacement (DLD) device that incorporates the power of neural network and multi-objective genetic algorithm.
## Design Automation
The design automation algorithms convert the user-defined specification to geometry properties and flow rate to readily meet the desired criteria. To boost the function evaluation speed, one can utilize neural networks as the surrogate model instead of the whole physics/experimentation. This alternative facilitates the design process of a DLD device to reach the best possible configuration, especially when combined with an evolutionary algorithm. As presented in Figure 2, design automation inputs are the diameters of particles to be separated (D1 and D2), desired constraints to be applied on f, N, Re (Cf , CN, and CRe), and finally the trade-off between flexibility and stability of the design (). Flexibility indicates the ability of the proposed design to cover a wide range of critical diameters while stability specifies its immunity to change due to minor flow rate fluctuations (see Appendix C). These two terms were included to consider the possible device behavior after fabrication since it can be altered by adjusting the Reynolds number.
After defining the inputs, a multi-objective optimization algorithm coupled with a pre-trained FCNN tunes the f, N, Re, and G to reach the optimized design. In this design automation tool, an NSGA3 with five reference directions and population size of 260 was chosen and implemented in Pymoo library [34] in Python 3. The pre-trained FCNN was used for direct critical diameter prediction (direct neural network) since extracting critical diameter from CNN outputs slows down the optimization process. Finally, the tuned parameters in addition to bandwidth (BW) are extracted and presented in the outputs. BW is the difference between the maximum and the minimum critical diameter for that design serving as an index of flexibility. Note that the parameter G is used in optimization to nondimensionalize the critical diameter matching the developed neural network’s Dc. Furthermore, the pre-trained CNN was attached to the end of the automation process to provide the ability to simulate the particle trajectory in the optimum design for a given number of periods. This feature helps visualize the particle behavior at the full length of the device.

For more information check: link will be added soon

![2-Design Automation](https://user-images.githubusercontent.com/97515569/179344205-92cddf73-6da5-44d6-9d3e-daed59f4f94e.png)

## Files Descriptions
###### utility functions 
- DLD_utils.py: Containing the function for extracting data from numerical simulation and mapping the generated field
- DLD_env.py: Containing the function for post processing the data from numerical simulation 
- particle_trajectory.py: simulate particle trajectory by having the velocity fields in horizontal and vertical direction

###### Convolutional Neural Network
- generate_data.py: Dataset generation for convolutional neural network
- Conv_base.py: The convolutional neural network class containng the network architecture
- Conv_net_train.py: The training process of Conculutional neural network
- Temp9: The file is not available in the repository it is accessable through this Link(https://drive.google.com/drive/folders/1--o_9SYRY1sq_FOjo_ogZG6z_9ejrwNq?usp=sharing)

###### Fully Connceted Neural Network 
- Direct_network_generate_data.py: Gets the labels to fileds dataset and change it into filed to critical diameter datset
- Direct_NN.py: The  fully conncted neural network class containing structure of the neural network, trainng process
- DNN_model_hlayers8_nodes_128.h5: The trained Direct neural network wights and bais 
###### Design Automation 
- Design_Aut.py: Main code of the design automation platform






