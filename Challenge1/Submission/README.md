# Submission Guide

**DO NOT COMMIT LARGE MODEL FILES TO GITLAB**

**ONLY CODE/NOTEBOOK/ETC TYPE FILES SHOULD BE UPLOADED HERE**

The files in this submission directory (and in /Submission/app) are designed
to quickly let you build a docker image that you can submit for challenges 1 and 2.

If you have not already done so, install Docker Desktop on your machine.

If you are unfamiliar with what Docker is, it represents a way to package
everything needed to run an application into a single object (a Docker image).

You will be building your own Docker image and submitting it to our Docker
registry for grading.

Everything needed to build a solution that randomly guesses (benign/malware) is provided.

Here is a list of the files in this Submission folder and why they matter:
* `Dockerfile` - These are the instructions that tell Docker how to build your image
* `requirements.txt` - Python libraries we'll install in the Docker image (not relevant if not using Python)
* `/app/my_inference_model.py` - Example Python model class that randomly guesses
* `/app/my_testing_script.py` - Example Python script for Docker to run.

Important note about these files:
`/app/my_testing_script.py` describes everything that your docker image needs to do.
If you don't want to use Python, feel free to adapt the steps in that script to your solution.

There are 2 main steps to submitting the model/solution you have on your machine
1) Build a Docker image (with your working code and models)
2) Push that Docker image to our Docker image registry

**To build your Docker image:**
1) In a terminal/cmd prompt, traverse to your Submission folder
(or wherever you have your code, models and Dockerfile)
2) Run `docker build -t my_cool_solution:latest .`
Note: the `.` at the end tells Docker that your Dockerfile is in the current dir

**To submit/push your Docker image:**
1) Login to our docker registry: `docker login registry.htm2021.boozallencsn.com`
2) Run `docker tag my_cool_solution:latest registry.htm2021.boozallencsn.com/hackthemachine/my_repo`
Note: `my_repo` will be replaced with your team's repo name (will be provided to you)
3) Run `docker push my_cool_solution`

As always, reach out in Slack if you run into issues and have questions.
