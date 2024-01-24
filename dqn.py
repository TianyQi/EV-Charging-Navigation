import networkx as nx
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import collections
from collections import deque
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from itertools import count

#update calculate reward based on electricity price and volume on the road
def weight_adj(G, price, alpha):
    for edge in list(G.edges):
        if G.edges[edge]['color'] == 'r':
            G.edges[edge]['weight'] = alpha * G.edges[edge]['distance']/(5 + np.random.randn() * 0.5)+(1-alpha) * price * G.edges[edge]['distance']/1000
        elif G.edges[edge]['color'] == 'y':
            G.edges[edge]['weight'] = alpha * G.edges[edge]['distance']/(10 + np.random.randn())+(1-alpha) * price * G.edges[edge]['distance']/1000
        elif G.edges[edge]['color'] == 'g':
            G.edges[edge]['weight'] = alpha * G.edges[edge]['distance']/(15 + np.random.randn()* 1.5)+(1-alpha) * price * G.edges[edge]['distance']/1000
    return G


#quantize the elec price
def quantize_price(price):
    return round(price*10)/10

# Define hyperparameters
BATCH_SIZE = 16
GAMMA = 0.99
EPS_START = 0.3
TARGET_UPDATE = 20
MEMORY_SIZE = 1000000
LR = 0.00001

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define replay memory buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        #batch = random.sample(self.memory, batch_size)
        #state, action, reward, next_state, done = map(torch.stack, zip(*batch))
        #return state, action, reward, next_state, done
        state, action, reward, next_state, done = zip(*random.sample(self.memory, batch_size))
        return np.stack(state), action, reward, np.stack(next_state), done

    def __len__(self):
        return len(self.memory)

# Define DQN model
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.fc1 = nn.Linear(input_shape, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = x.view(-1, 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Define epsilon-greedy policy
def select_action(state, policy_net):
    action_space = [0,1,2]
    sample = random.random()
    eps_threshold = EPS_START 
    if sample > eps_threshold:
        with torch.no_grad():
            #print(state)
            #print(policy_net(state.to(device)))
            #print(policy_net(state.to(device)).max(1)[1].item())
            #return policy_net(state.to(device)).max(1)[1].item()
            return policy_net(state.to(device)).max(1)[1].item() #损耗最小
    else:
        return random.sample(action_space,1)[0]

# Define training function
def train(policy_net, target_net, optimizer, memory):
    if len(memory) < BATCH_SIZE:
        return
    state, action, reward, next_state, done = memory.sample(BATCH_SIZE)

    state = torch.FloatTensor(np.float32(state)).to(device)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    next_state = torch.FloatTensor(np.float32(next_state)).to(device)
    done = torch.FloatTensor(done).to(device) 

    #state_action_values = policy_net(state).gather(1, action)
    state_action_values = policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1)
    #next_state_values = torch.zeros(BATCH_SIZE, device=device)
    #next_state_values[~done] = target_net(next_state[~done]).max(1)[0].detach()
    #expected_state_action_values = (next_state_values * GAMMA) + reward
    next_q_value = target_net(next_state).max(1)[0].detach() #
    expected_state_action_values = reward + GAMMA * next_q_value * (1 - done)
    #loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    loss = F.mse_loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#save DQN model
def save(model,filename):
    torch.save(model.state_dict(), filename)

#test the performance of the model
def test(G, model, reward_record):
    #copy the graph
    G_4 = G
    G_22 = G
    G_32 = G
    start_node = 11
     # select an action and take a step,first generate 3 real-time elec price
    price_4 = 0.5 + np.random.randn() * 0.15
    price_22 = 0.5 + np.random.randn() * 0.15
    price_32 = 0.5 + np.random.randn() * 0.15

    price_4 = quantize_price(price_4)
    price_22 = quantize_price(price_22)
    price_32 = quantize_price(price_32)
    #update each graph
    G_4 = weight_adj(G_4, price_4, alpha=0.3)
    G_22 = weight_adj(G_22, price_22, alpha=0.3)
    G_32 = weight_adj(G_32, price_32, alpha=0.3)
    #calculate each graph's cost
    reward_record = 0
    steps_done = 0
    state = torch.FloatTensor([start_node, price_4, price_22, price_32]) # start node
    #select a EVCS based on Q-network
    action = select_action(state, model)
    while(1):
            
        # select an action and take a step,first generate 3 real-time elec price
        price_4 = 0.5 + np.random.randn() * 0.15
        price_22 = 0.5 + np.random.randn() * 0.15
        price_32 = 0.5 + np.random.randn() * 0.15

        price_4 = quantize_price(price_4)
        price_22 = quantize_price(price_22)
        price_32 = quantize_price(price_32)
        #update each graph
        G_4 = weight_adj(G_4, price_4, alpha=0.3)
        G_22 = weight_adj(G_22, price_22, alpha=0.3)
        G_32 = weight_adj(G_32, price_32, alpha=0.3)
        #calculate each graph's cost

        #select a EVCS based on Q-network
        action = select_action(state, policy_net)

        #calculate reward, next_state and done
        #print(int(state[0].item()))
        if action == 0:
            if int(state[0].item()) == 4:
                done = 1
                break
            G_4 = weight_adj(G_4, price_4, alpha=0.3)
            next_state = torch.tensor([list(nx.dijkstra_path(G_4,int(state[0].item()),4))[1], price_4, price_22, price_32])
            reward = -nx.dijkstra_path_length(G_4, int(state[0].item()), int(next_state[0].item()))
                
        elif action == 1 :
            if int(state[0].item()) == 22:
                done = 1
                break
            G_22 = weight_adj(G_22, price_22, alpha=0.3)
            next_state = torch.tensor([list(nx.dijkstra_path(G_22,int(state[0].item()),22))[1], price_4, price_22, price_32])
            reward = -nx.dijkstra_path_length(G_22, int(state[0].item()), int(next_state[0].item()))
                
        else :
            if int(state[0].item()) == 32:
                done = 1
                break
            G_32 = weight_adj(G_32, price_32, alpha=0.3)
            next_state = torch.tensor([list(nx.dijkstra_path(G_32,int(state[0].item()),32))[1], price_4, price_22, price_32])
            reward = -nx.dijkstra_path_length(G_32, int(state[0].item()), int(next_state[0].item()))
                
        #print(int(state[0].item()))
        
        reward_record += reward
        #print(reward)
        state = next_state
        steps_done += 1

    return reward_record

if __name__ == "__main__":

    # Create an empty graph object
    G = nx.Graph()

    # Add 39 nodes to the graph
    for i in range(1,40):
        G.add_node(i)

    # Add edges to the graph with origin weights, colors and distance(m)
    G.add_edge(1, 2, weight=1, color='g',distance=1000)
    G.add_edge(1, 5, weight=1, color='y',distance=1000)
    G.add_edge(1, 3, weight=1, color='g',distance=1500)
    G.add_edge(2, 4, weight=1, color='y',distance=800)
    G.add_edge(2, 13, weight=1, color='g',distance=2000)
    G.add_edge(3, 7, weight=1, color='y',distance=1300)
    G.add_edge(3, 8, weight=1, color='g',distance=900)
    G.add_edge(4, 5, weight=1, color='r',distance=200)
    G.add_edge(4, 14, weight=1, color='y',distance=1000)
    G.add_edge(5, 6, weight=1, color='r',distance=600)
    G.add_edge(6, 7, weight=1, color='r',distance=400)
    G.add_edge(6, 15, weight=1, color='r',distance=1000)
    G.add_edge(7, 9, weight=1, color='r',distance=900)
    G.add_edge(8, 9, weight=1, color='y',distance=1400)
    G.add_edge(8, 10, weight=1, color='y',distance=1400)
    G.add_edge(8, 11, weight=1, color='g',distance=2600)
    G.add_edge(9, 10, weight=1, color='r',distance=1000)
    G.add_edge(9, 16, weight=1, color='r',distance=900)
    G.add_edge(10, 12, weight=1, color='y',distance=1400)
    G.add_edge(10, 17, weight=1, color='r',distance=1000)
    G.add_edge(11, 12, weight=1, color='g',distance=1000)
    G.add_edge(12, 18, weight=1, color='g',distance=1200)
    G.add_edge(13, 14, weight=1, color='y',distance=1100)
    G.add_edge(13, 19, weight=1, color='y',distance=400)
    G.add_edge(13, 24, weight=1, color='g',distance=1100)
    G.add_edge(14, 15, weight=1, color='y',distance=900)
    G.add_edge(15, 16, weight=1, color='r',distance=1300)
    G.add_edge(15, 20, weight=1, color='r',distance=900)
    G.add_edge(16, 17, weight=1, color='r',distance=1000)
    G.add_edge(16, 21, weight=1, color='r',distance=1200)
    G.add_edge(17, 18, weight=1, color='y',distance=1500)
    G.add_edge(17, 22, weight=1, color='r',distance=1300)
    G.add_edge(18, 30, weight=1, color='g',distance=2500)
    G.add_edge(18, 23, weight=1, color='y',distance=1500)
    G.add_edge(19, 20, weight=1, color='y',distance=1300)
    G.add_edge(19, 25, weight=1, color='y',distance=1000)
    G.add_edge(20, 21, weight=1, color='r',distance=1400)
    G.add_edge(20, 26, weight=1, color='r',distance=1000)
    G.add_edge(21, 22, weight=1, color='r',distance=900)
    G.add_edge(21, 27, weight=1, color='r',distance=900)
    G.add_edge(22, 23, weight=1, color='y',distance=500)
    G.add_edge(22, 29, weight=1, color='r',distance=600)
    G.add_edge(23, 30, weight=1, color='y',distance=1200)
    G.add_edge(24, 25, weight=1, color='y',distance=300)
    G.add_edge(24, 37, weight=1, color='g',distance=3000)
    G.add_edge(25, 26, weight=1, color='y',distance=900)
    G.add_edge(26, 27, weight=1, color='r',distance=1600)
    G.add_edge(26, 32, weight=1, color='r',distance=900)
    G.add_edge(27, 28, weight=1, color='r',distance=300)
    G.add_edge(27, 33, weight=1, color='r',distance=1000)
    G.add_edge(27, 34, weight=1, color='r',distance=800)
    G.add_edge(28, 29, weight=1, color='r',distance=200)
    G.add_edge(28, 35, weight=1, color='r',distance=1000)
    G.add_edge(29, 31, weight=1, color='y',distance=2300)
    G.add_edge(30, 31, weight=1, color='g',distance=400)
    G.add_edge(32, 33, weight=1, color='y',distance=1000)
    G.add_edge(32, 37, weight=1, color='y',distance=1400)
    G.add_edge(33, 34, weight=1, color='y',distance=1400)
    G.add_edge(33, 38, weight=1, color='y',distance=1300)
    G.add_edge(34, 35, weight=1, color='y',distance=800)
    G.add_edge(34, 39, weight=1, color='y',distance=1700)
    G.add_edge(35, 36, weight=1, color='y',distance=900)
    G.add_edge(36, 39, weight=1, color='g',distance=1200)
    G.add_edge(37, 38, weight=1, color='g',distance=1300)
    G.add_edge(38, 39, weight=1, color='g',distance=2300)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # choose to go to 1 of 3 EVCSs, num od EVCSs are 4, 22, 32 indicate 0,1,2
    n_actions = 3
    action_space = np.array([0,1,2])

    #state is the num of the node and the current electricity price
    state_shape = 4

    #generate a start node
    start_node = np.random.randint(1,40)

    #copy the graph
    G_4 = G
    G_22 = G
    G_32 = G

    #define DQN parameter
    memory = ReplayMemory(MEMORY_SIZE)
    policy_net = DQN(state_shape, n_actions).to(device)
    target_net = DQN(state_shape, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    steps_done = 0
    rewards = []
    record_reward = 0
    record_test = []

    # Train the model
    num_episodes = 100
    for i_episode in range(num_episodes):
        
        done = False
        total_reward = 0
        start_node = np.random.randint(1,40)
        #start_node = 38
        price_4 = 0
        price_22 = 0
        price_32 = 0
        state = torch.FloatTensor([start_node, price_4, price_22, price_32]) # start node
        #print(state[0].dtype)

        while(1):
            if (int(state[0].item()) == 4) or (int(state[0].item()) == 22) or (int(state[0].item()) == 32) :
                done = 1
                print(int(state[0].item()))
                break
            # select an action and take a step,first generate 3 real-time elec price
            price_4 = 0.5 + np.random.randn() * 0.15
            price_22 = 0.5 + np.random.randn() * 0.15
            price_32 = 0.5 + np.random.randn() * 0.15

            price_4 = quantize_price(price_4)
            price_22 = quantize_price(price_22)
            price_32 = quantize_price(price_32)
            #update each graph
            G_4 = weight_adj(G_4, price_4, alpha=0.3)
            G_22 = weight_adj(G_22, price_22, alpha=0.3)
            G_32 = weight_adj(G_32, price_32, alpha=0.3)
            #calculate each graph's cost

            #select a EVCS based on Q-network
            action = select_action(state, policy_net)
            print(state)
            print(policy_net(state))
            print(policy_net(state).max(1)[0])
            print(policy_net(state).max(1)[1])
            #print(policy_net(state).gather(1, action.unsqueeze(1)).squeeze(1))
            print(state[0].item(),action)
            #calculate reward, next_state and done
            #print(int(state[0].item()))
            if action == 0:
                if int(state[0].item()) == 4:
                    done = 1
                    break
                G_4 = weight_adj(G_4, price_4, alpha=0.3)
                next_state = torch.tensor([list(nx.dijkstra_path(G_4,int(state[0].item()),4))[1], price_4, price_22, price_32])
                reward = nx.dijkstra_path_length(G_4, int(state[0].item()), int(next_state[0].item()))
                
            elif action == 1 :
                if int(state[0].item()) == 22:
                    done = 1
                    break
                G_22 = weight_adj(G_22, price_22, alpha=0.3)
                next_state = torch.tensor([list(nx.dijkstra_path(G_22,int(state[0].item()),22))[1], price_4, price_22, price_32])
                reward = nx.dijkstra_path_length(G_22, int(state[0].item()), int(next_state[0].item()))
                
            else :
                if int(state[0].item()) == 32:
                    done = 1
                    break
                G_32 = weight_adj(G_32, price_32, alpha=0.3)
                next_state = torch.tensor([list(nx.dijkstra_path(G_32,int(state[0].item()),32))[1], price_4, price_22, price_32])
                reward = nx.dijkstra_path_length(G_32, int(state[0].item()), int(next_state[0].item()))
                
            #print(int(state[0].item()))
            reward = -reward
            total_reward += reward
            #print(reward)
            reward = torch.tensor([reward], device=device)
            #memory.push(torch.tensor([state], device=device, dtype=torch.float32), action, reward, torch.tensor([next_state], device=device, dtype=torch.float32), torch.tensor([done], device=device, dtype=torch.uint8))
            memory.push(state, action, reward, next_state, done)
            state = next_state
            steps_done += 1
            
            train(policy_net, target_net, optimizer, memory)
            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())
                save(policy_net, 'EV-NAVI.pt')
            if done:
                break
        
        if i_episode % 1 == 0:
            print('Episode: {}, Total Reward: {}'.format(i_episode, total_reward))
            #print(agent.rewards)
        
        if i_episode % 10 == 0:
            temp = 0
            for j in range(20):
                temp += test(G, policy_net, record_reward)
            record_test.append(temp/20)

        rewards.append(total_reward)

    def move_ave(mat,stride):
        new = torch.zeros(len(mat))
        for i in range(len(mat)):
            if(i<(len(mat)-stride)):
                new[i] = torch.mean(mat[i:i+stride])
            else:
                new[i] = torch.mean(mat[i:len(mat)-1])
        return new
    
    print(record_test)
    x = torch.arange(1, num_episodes/5+1)*5
    record_test = torch.tensor(record_test)
    #rewards = torch.tensor(rewards)
    #reward_ave = move_ave(rewards, 20)
    plt.plot(x, record_test,color = 'c')
    #plt.plot(x, reward_ave, color = 'b')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Testing Results per 5 training round')
    plt.savefig('EV-NAVI-test reward.png')

        
    
