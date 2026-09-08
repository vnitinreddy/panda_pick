#Panda Pick

This project uses OpenVLA and MuJoCo to pick an object!

The code is tested on Google Colab using A100 GPU. Both MuJoCo simulation and OpenVLA model are running in Colab.

Initially, I started with MuJoCo running on my laptop and OpenVLA running on Google Colab. The delay was moving the hand slowly and
hence switched to running everything on Colab.

The challenge with running everything on Colab is to make sure environment is setup corredtly as Colab notebook comes with a default
environment and then making sure we rely on offline rendering as Colab is headless and we cannot render as we move the hand (Instead,
we save images and replay all of them later!).
