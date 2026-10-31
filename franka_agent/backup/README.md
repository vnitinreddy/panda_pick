#Panda Pick

bridge_to_brain*.py: The client that runs on local machine. It runs the MuJoCo simulation, connects to the OpenVLA running in Google Colab, and controls the simulated Panda arm based on predictions from openVLA server. You need to run the server (open_vla_only_on_colab.ipynb) in Google Colab before starting the client on the local machine. 

The other open_vla_and_mujoco_*.ipynb are variations of notebooks used when testing both the OpenVLA and MuJoCo running in Google Colab.
