#!/usr/bin/env python
from environments import environment_setup, DogAdventure, DogAdventureEnvironment
from train import Trainer
from evaluate import Evaluator
from visualize import Progress_viz

if __name__ == '__main__':

    print("Step 1: Environment Setup")
    dogEnvironment = DogAdventureEnvironment(DogAdventure())
    train_env, eval_env = environment_setup(dogEnvironment)

    print("Step 2: Trainer Setup")
    dogTrainer = Trainer(train_env, n_iterations=15, visual_flag=True)

    # generate training trajectories
    dogTrainer.data_generation()

    # run under common to improve efficiency
    dogTrainer.make_common()

    print("Step 3: Train the Model")
    metrics, losses = dogTrainer.train_agent()

    print("Step 4: Evaluate Learning Result")
    # Reset the train step
    dogTrainer._agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    dogEvaluator = Evaluator(eval_env, dogTrainer._agent, dogTrainer._replay_buffer, dogTrainer._train_step, episodes=1, visual_flag=True)
    #dogEvaluator.evaluate_agent()

    print("Step 5: Checkpoint Saver")
    dogEvaluator.save_model()