# MODSIM: an efficent Marine Object Dection Simulator


This repository contains the system developed for a master's thesis in Computer Science at the Norwegian University of Science and Technology (NTNU). The system's components are described in detail in the thesis.


### Error generation

To accurately simulate the errors made by the desired detector, you need to define specific error metrics. These metrics should be provided in a YAML file. We have provided an example YAML file at `src/detector_stats_config.yaml`.

### Required Parameters
The following parameters are required for the simulation:

- Possible labels: The possible labels the detector can classify the detected objects as.

- Number of states: The desired number of temporal states in the model.

- Initial state: The initial state of the temporal model.

- Transition matrix: A matrix that specifies the probabilities of transitioning between the temporal states.

- Mean vector: The mean vector, i.e. expected value vector, of the bounding box error vectors. One for each finite state in the temporal model.

- Covariance matrix: The covariance matrix of the bounding box error vectors. One for each finite state in the temporal model.

- Confidence threshold: A pre-defined minimum level of confidence required for a detection to be outputted by the detector.

- Probabilistic confusion matrices: One for each finite state in the temporal model reflecting the desired performance of the simulated detector under different detection conditions.

- False discovery rate: One for each finite state in the temporal model reflecting the desired false discovery rate of the simulated detector under different detection conditions.


### Simulations with pose data generated with a circular motion model
Simulations can be created using the circular motion model by running the following file:

´´´
src/MODSIM_w_dsg.ipynb
´´´

### Simulations reading predefined pose data
Simulations can be created using predefined pose data by running the following fil: 

´´´
src/MODSIM_w_pose_data.ipynb
´´´
