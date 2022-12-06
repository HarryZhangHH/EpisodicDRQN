# PartnerSelection
Implementation for the paper: Partner Selection for Emergence of cooperation in multi-agent systems using reinforcement learning [^1]

# Setting up an environment
create an environment with the command:
```
conda env create -f path/to/environment.yml
```

Removing the environment if something goes wrong
```
conda remove --name partner_selection --all
```
Activating / Deactivating the environment
```
conda activate partner_selection

conda deactivate partner_selection
```

# Sample Execution 
If run with default command line arguments, using

```
python main.py
```

If check the command line arguments explanation, using 

```
python main.py -h
```

Example command line arguments, using

```
python main.py --h=10 --state_repr=bi --n_episode=10000 --batch_size=64
```
## When running
After you successfully run the code, you will have 4 choices. <br />
Choice **0** is to run the 2-agents benchmark. <br />
Choice **1** is to test a method with all the other other method in 2-agents setting. <br />
Choice **2** is to test a reinforcement learning method with another specific method in 2-agents setting. <br />
Choice **3** is to run the multi-agents game. 

### The selectable methods: 
```
choices = {'0-alwaysCooperate','1-alwaysDefect','2-titForTat','3-reverseTitForTat','4-random','5-grudger','6-pavlov','7-qLearning','8-lstm-pavlov','9-dqn','10-lstmqn','11-a2c','12-a2c-lstm'}
rl_choices = {'7-qLearning','8-lstm-pavlov','9-dqn','10-lstmqn','11-a2c','12-a2c-lstm'}
```
The first 7 methods are the fix-strategy methods.

7-qLearning is the tabular q learning method, you can change the `--h=` and `--state_repr=` in command line arguments to determine the tabular size. Note that using tabular method, the maximize of h is 3 and you can only use the label based state representation.

8-lstm-pavlov is a LSTM[^3] method using the LSTM network to predict the next action of your opponent, you can change the `--h=` to determine the sequence length. Note that you can not use the label based state representation in the lstm based method. I personally recommend `--state_repr=bi`.

9-dqn is a DQN[^2] method to predict the Q value.

10-lstmqn is a DQN[^2] method using the LSTM[^3] network to predict the Q value. Note that you can not use the label based state representation in the lstm based method. I personally recommend `--state_repr=bi`.

11-a2c is a ActorCritic[^4] method using the deep NN. You may need to manually change the `Class Worker` in the `./agent/actor_critic_agent.py` to your preference of worker number and worker against strategy.

12-a2clstm is a ActorCritic[^4] method using the LSTM. You may need to manually change the `Class Worker` in the `./agent/actor_critic_lstm_agent.py` to your preference of worker number and worker against strategy.

Note that, you can manually change some hyperparameters (like `HIDDEN_SIZE`, `TARGET_UPDATE`) in the specific .py file.


### The state representation: 
```
choices = {'uni', 'bi', 'unilabel', 'grudgerlabel', 'bi-repr'}
```
'uni' is only

[^1]: [Partner Selection for Emergence of cooperation in multi-agent systems using reinforcement learning](https://arxiv.org/abs/1902.03185)
[^2]: [DQN](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
[^3]: [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) 
[^4]: [Actor Critic](https://omegastick.github.io/2018/06/25/easy-a2c.html)
