from __future__ import absolute_import
from __future__ import print_function


import datetime
from training_simulation import Simulation
from generator import TrafficGenerator
from replay_buffer import ReplayBuffer
from model import DQN
from utils import import_train_configuration, set_sumo, set_train_path
import wandb

if __name__ == "__main__":

    # import config and init config
    config = import_train_configuration(config_file='settings/training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    DQN = DQN(
        config['width_layers'],
        input_dim=config['num_states'], 
        output_dim=config['num_actions']
    )

    ReplyBuffer = ReplayBuffer(
        config['memory_size_max'], 
        config['memory_size_min']
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated']
    )
        
    Simulation = Simulation(
        DQN,
        ReplyBuffer,
        TrafficGen,
        sumo_cmd,
        config['gamma'],
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration'],
        config['num_states'],
        config['num_actions'],
        config['training_epochs'],
        config['batch_size'],
        config['learning_rate']
    )
    
    episode = 0
    timestamp_start = datetime.datetime.now()

    #project = "DQN ATL"
    #wandb.init(project=project)

    while episode < config['total_episodes']:
        print('\n [INFO]----- Episode', str(episode + 1), '/', str(config['total_episodes']), '-----')
        # set the epsilon for this episode according to epsilon-greedy policy
        epsilon = 1.0 - (episode / config['total_episodes'])
        # run the simulation
        simulation_time, training_time, avg_reward, training_loss = Simulation.run(episode, epsilon)
        print('\t [STAT] Simulation time:', simulation_time, 's - Training time:',
              training_time, 's - Total:', round(simulation_time + training_time, 1), 's')
        # log the training progress in wandb
        #wandb.log({
            #"all/training_loss": training_loss,
            #"all/avg_reward": avg_reward,
            #"all/avg_waiting_time": avg_waiting,
            #"all/simulation_time": simulation_time,
            #"all/training_time": training_time,
            #"all/entropy": epsilon}, step=episode)
        episode += 1
        print('\t [INFO] Saving the model')
        Simulation.save_model(path, episode)

    print("\n [INFO] End of Training")
    print("\t [STAT] Start time:", timestamp_start)
    print("\t [STAT] End time:", datetime.datetime.now())
    print("\t [STAT] Session info saved at:", path)
