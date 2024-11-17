# ECOLIFE: Carbon-Aware Serverless Function Scheduling for Sustainable Computing 

## Table of Contents
- [About the Project](#about-the-project)
- [Setup](#setup)
- [How to run](#how_to_run)
- [Tasks](#tasks)


## About the Project
In this paper, we present EcoLife, an innovative strategy that leverages multi-generation hardware to co-optimize carbon footprint and service time within the serverless environment. EcoLife extends Particle Swarm Optimization (PSO) to adapt to the variations in serverless computing for making keep-alive and execution decisions. Our experimental results show that EcoLife effectively reduces carbon emissions while maintaining high performance for function execution.



## Setup
- **Language**: Python3.10
- **Hardware**: This repo is mainly simulation, you don't need any hardware to reproduce the results.
- **Libs**: Ensure that all the libs are installed.

## How to Run:
```
│eco-life/
  ├──carbon_intensity/
  ├──data/
  ├──motivations/
  ├──node/
  ├──optimizers/
  ├──results/
```
1. `carbon_intensity` contains the carbon intensity for various regions, Use one of them to simulate.
2. `data` contains the profiled data for optimization. (eg. the carbon and energy data for different serverless functions)
3. `motivations`: 4 motivations in the paper.
4. `node`: Generate the profiled data. You don't need this.
5. `optimizers`: different optimizers in Ecolife
6. `results`: you may need to save your results in this folder.
7. `exe_decide.py`:Execution Placement Decision Maker.
8. `function_mem.csv`: Memory consumption of different serverless functions.
9. `main.py`: You may use it to run the codebase.
10. `pso.py`: DPSO in Ecolife. You need to change it.
11. `utils.py`: Help functions.
12. `selected_trace.zip`: Traces for simulation.

### Run:
```bash
unzip selected_trace.zip
python3 main.py <add your desired configuration>
```
When you run the main.py, check the code.



## Tasks

Here are some tasks for you.
- Read the paper. [link](https://arxiv.org/pdf/2409.02085)
- Read the code.
- Weakness/improvement of this paper.
- **Change DPSO to other heuristic algorithms**. If you don't know what's a heuristic algorithm, check this: link(https://optimization.cbe.cornell.edu/index.php?title=Heuristic_algorithms)
  Note that, you may need to **compare the results with DPSO** in Ecolife. **Plotting a bar figure** is a good way to try.
- Don't use 12 days for simulation, it will take 2-3 days. **Just use one day**.
- If you have any questions, contact me! jiang.yank@northeastern.edu



