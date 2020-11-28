#!/usr/bin/env python
import param
from train import Trainer
from evaluate import Evaluator
from visualize import Progress_viz
from environments import environment_setup, CamAdventureGame, CamAdventureEnvironment

if __name__ == '__main__':

    print("Step 1: Environment Setup")
    camEnvironment = CamAdventureEnvironment(CamAdventureGame())
    train_env, eval_env = environment_setup(camEnvironment)
    

    print("Step 2: Trainer Setup")
    camTrainer = Trainer(train_env, n_iterations=param.TRAIN_ITER, visual_flag=param.VIZ_FLAG)

    # generate training trajectories
    camTrainer.data_generation()

    # Run under common to improve efficiency
    camTrainer.make_common()

    print("Step 3: Train the Model")
    # reset the train step
    camTrainer._agent.train_step_counter.assign(0) 

    # start training the model
    metrics, losses = camTrainer.train_agent()

    print("Step 4: Evaluate Learning Result")
    # Evaluate the agent's policy
    camEvaluator = Evaluator(eval_env, camTrainer._agent, camTrainer._replay_buffer, 
                             camTrainer._train_step, episodes=param.EVAL_EPISODE, 
                             visual_flag=param.VIZ_FLAG)
    camEvaluator.evaluate_agent()

    print("Step 5: Checkpoint Saver")
    camEvaluator.save_model()