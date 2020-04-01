from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook
import imageio,pickle


class DeliveryEnvironment:

    def __init__(self, n_carriers, n_stores, n_customers, max_radius=10):
        self.n_carriers = np.array(n_carriers)
        self.n_stores = np.array(n_stores)
        self.n_customers = n_customers
        self.max_radius = max_radius
        self.stops = []
        self.customers = []
        self.reached_points=[]

        self.max_custs = dict()
        self.curr_custs = dict()

        self.get_positions()


        self.action_space = len(self.n_carriers) + len(self.n_stores) + len(self.customers)

        self.generate_q_values()
        self.render()

    def get_positions(self):
        self.customers.extend(self.n_customers[0])
        self.customers.extend(self.n_customers[1])
        self.customers.extend(self.n_customers[2])
        self.customers = np.array(self.customers)

        xy = list()
        xy.extend(self.n_carriers)
        xy.extend(self.n_stores)
        xy.extend(self.customers)

        xy = np.array(xy)
        self.x = xy[:, 0]
        self.y = xy[:, 1]

    def reset_reached_points(self):
        self.reached_points = np.zeros(self.action_space)

    def generate_q_values(self):
        # xy = np.column_stack([self.x, self.y])
        # self.q_stops = cdist(xy, xy)
        start_q_table = 'qtable.pickle'
        if start_q_table is None:
            # initialize the q-table#
            xy = np.column_stack([self.x, self.y])
            self.q_stops = cdist(xy, xy)
        else:
            with open(start_q_table, "rb") as f:
                self.q_stops = -1*pickle.load(f)

    def render(self, return_img=False):
        colors = ['g', 'r', 'c', 'm', 'y', 'k', 'w']

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.set_title('Delivery stops')

        ax.scatter(self.n_carriers[:, 0], self.n_carriers[:, 1], c='b', s=50)
        for i in range(len(self.n_stores)):
            ax.scatter(self.n_stores[i, 0], self.n_stores[i, 1], c=colors[i], s=50, marker='P')
            ax.scatter(np.array(self.n_customers[i])[:, 0], np.array(self.n_customers[i])[:, 1], c=colors[i], s=50,
                        marker='*')

        # print(self.stops)
        if len(self.stops) > 0:
            xy = self._get_xy(initial=True)
            xytext = xy[0] + 0.1, xy[1] - 0.05
            ax.annotate("START", xy=xy, xytext=xytext, weight="bold")

        if len(self.stops) > 1:
            ax.plot(self.x[self.stops], self.y[self.stops], c="blue", linewidth=1, linestyle="--")

            # Annotate END
            xy = self._get_xy(initial=False)
            xytext = xy[0] + 0.1, xy[1] - 0.05
            ax.annotate("END", xy=xy, xytext=xytext, weight="bold")

        plt.xticks([])
        plt.yticks([])

        if return_img:
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image
        # else:
        #     plt.show()

    def _get_xy(self,initial = False):
        state = self.stops[0] if initial else self._get_state()
        x = self.x[state]
        y = self.y[state]
        return x,y

    def reset(self, first_stop):
        self.stops = []
        self.stops.append(first_stop)
        self.reached_points[first_stop] = 1
        return first_stop

    def step(self, destination):
        state = self.stops[-1]
        new_state = destination
        reward=0
        # print('reached')

        if len(self.n_carriers) <= destination < len(self.n_stores) + len(self.n_carriers):
            customer_x = np.array(self.n_customers[destination - len(self.n_carriers)])[:, 0]
            customer_index = []
            for m in customer_x:
                idx = np.where(self.x == m)
                customer_index.append(list(idx)[0][0])
            flag = True
            for x in customer_index:
                if self.reached_points[x] == 1:
                    flag = True
                else:
                    flag = False
                    break
            if flag == True and destination in self.stops:
                self.reached_points[destination] = 1
            else:
                reward = self._get_reward(state, new_state)
                self.stops.append(destination)
                self.reached_points[destination] = 0
        elif destination >= len(self.n_stores) + len(self.n_carriers):
            key = -1
            for cust_list in list(self.n_customers.values()):
                if self.x[destination] in np.array(cust_list)[:, 0]:
                    key = list(self.n_customers.values()).index(cust_list)
                    break
            if key >= 0 and (key + len(self.n_carriers)) in self.stops:
                reward = self._get_reward(state, new_state)
                self.stops.append(destination)
                self.reached_points[destination] = 1
                flag = True
                for val in self.n_customers[key]:
                    m = np.where(self.customers == val)
                    x = m[0][0]
                    idx = x + len(self.n_stores) + len(self.n_carriers)
                    if self.reached_points[idx] == 1:
                        flag = flag and True
                    else:
                        flag = flag and False
                        break

                if flag:
                    # print(self.stops,key + len(self.n_carriers))
                    self.reached_points[key + len(self.n_carriers)] = 1
                else:
                    self.reached_points[key + len(self.n_carriers)] = 0
        # else:
        #     reward = self._get_reward(state, new_state)
        #     self.stops.append(destination)
        #     self.reached_points[destination] = 1
        # print(self.stops,self.reached_points)

        carr_done = len([x for x in range(len(self.n_carriers)) if self.reached_points[x] == 1])
        cust_done = len([x for x in self.stops if x > len(self.n_carriers)+len(self.n_stores) and self.reached_points[x] == 1])
        min_cust = len(self.n_stores)
        max_cust = (len(self.customers)//len(self.n_carriers))
        done=False
        if len(self.n_carriers)-carr_done == 0:
            done = True if np.all(self.reached_points[len(self.n_carriers):]) else False
            # print(self.stops,destination,done)
        else:
            if max_cust > cust_done > min_cust:
                true_val = 0.1
                # print(true_val,cust_done/min_cust)
                if true_val>1:
                    true_val = 1
                done = True if np.all(
                    self.reached_points[len(self.n_carriers):]) else self.customer_to_store_check() and np.random.choice(
                    [True, False], p=[1-true_val, true_val])
            elif max_cust <= cust_done:
                done = True and self.customer_to_store_check()
            else:
                done=False


        return new_state, reward, done

    def _get_state(self):
        return self.stops[-1]

    def _get_xy(self, initial=False):
        state = self.stops[0] if initial else self._get_state()
        x = self.x[state]
        y = self.y[state]
        return x, y

    def customer_to_store_check(self):
        flag = True
        for x in self.stops:
            if len(self.n_carriers) <= x < len(self.n_carriers) + len(self.n_stores):
                customer_x = np.array(self.n_customers[x - len(self.n_carriers)])[:, 0]
                flag1 = False
                for each in self.stops[self.stops.index(x):]:
                    if self.x[each] in customer_x:
                        flag1 = True
                flag = flag and flag1
            # print(x,flag)
        return flag

    def set_max_custs(self, new_state):
        if new_state not in list(self.max_custs.keys()):
            self.max_custs[new_state] = np.random.randint(1,
                                                          len(self.n_customers[new_state - len(self.n_carriers)]) + 1)

    def next_check(self, state, new_state):
        # carrier, store
        if self.reached_points[new_state]==1:
            return False
        if state < len(self.n_carriers) <= new_state < len(self.n_stores) + len(self.n_carriers):
            return True

        # store, store
        elif len(self.n_carriers) <= state < len(self.n_stores) + len(self.n_carriers) and len(
                self.n_carriers) <= new_state < len(
            self.n_stores) + len(self.n_carriers):
            return True

        # store/customer,customer
        elif new_state >= len(self.n_stores) + len(self.n_carriers) and state >= len(self.n_carriers):
            ordered_store = self.get_store_of_customer([self.x[new_state], self.y[new_state]]) + len(self.n_carriers)
            if ordered_store in self.stops:
                return True
            else:
                return False

        # customer,store
        elif state >= len(self.n_stores) + len(self.n_carriers) > new_state >= len(self.n_carriers):
            return True

        # all other
        else:
            return False

    def get_store_of_customer(self, customer):
        key = -1
        for k in list(self.n_customers.keys()):
            vals = self.n_customers[k]
            if customer in vals:
                key = k
                break
        return key

    def _get_reward(self, state, new_state):
        base_reward = self.q_stops[state, new_state]
        return base_reward



class DeliveryQAgent:

    def __init__(self, env, states_size, actions_size, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999, gamma=0.95,
                 lr=0.8):
        self.env = env
        self.states_size = states_size
        self.actions_size = actions_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.Q = self.build_model()

    def build_model(self):
        Q = np.zeros([self.states_size, self.actions_size])
        return Q

    def train(self, s, a, r, s_next):
        self.Q[s, a] = self.Q[s, a] + self.lr * (r + self.gamma * np.max(self.Q[s_next, a]) - self.Q[s, a])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def action(self, s):
        q = np.copy(self.Q[s, :])
        q[0:len(self.env.n_carriers)]= -np.inf
        q[self.states_memory] = -np.inf
        # print('Q')

        if np.random.rand() > self.epsilon:
            a = np.argmax(q)
            # print('yes')
        else:
            li = [x for x in range(self.actions_size) if x not in self.states_memory and self.env.next_check(s, x)]
            # print('li',li,self.env.stops,self.env.reached_points,self.states_memory)
            if len(li) > 0:
                a = np.random.choice(li)
            else:
                a = -99
        # print('a',a)

        return a

    def remember_state(self, s):
        self.states_memory.append(s)

    def reset_memory(self):
        self.states_memory = []



def run_task(env, agent, first_stop=0):
    s = env.reset(first_stop)
    agent.reset_memory()
    episode_reward = 0
    stops=[first_stop]
    i = 0
    while not np.all(env.reached_points[len(env.n_carriers):]):
        agent.remember_state(env.stops[-1])
        # Choose an action
        a = agent.action(s)
        # Take the action, and get the reward from environment
        done = False
        if a==-99:
            break
        if env.reached_points[a] == 0:
            s_next, r, done = env.step(a)
            # Tweak the reward
            r = -1 * r
            # Update our knowledge in the Q-table
            agent.train(s, a, r, s_next)
            # Update the caches
            episode_reward += r
            s = s_next
            # if s_next<3:
            #     print('$$$$$$$$$$$$$$$$$'
            #           ,s_next,a)
            stops.append(s_next)
            # print(stops)

        i += 1
        if done:
            # print('\nStops:\n', stops,env.stops)
            # print(env.reached_points)
            break

    return env.stops, episode_reward



carriers = [[3.69762835, 7.57409778], [9.03261303, 0.22846886],[3.53768079, 5.87194668]]
stores = [[9.53524493, 0.36075049], [3.53768079, 5.87194668], [0.93675762, 0.17497906]]
customers = {0: [[3.80095545, 5.52088867], [7.63991365, 2.36346141],[9.93261303, 0.12846886],[3.53768079, 5.87194668]],
             1: [[2.75104152, 4.9312129], [1.60984865, 7.91871507]],
             2: [[9.22365935, 2.32576777], [6.85037247, 5.61773748],[6.69762835, 3.57409778]]}

env = DeliveryEnvironment(carriers, stores, customers)
agent = DeliveryQAgent(env, env.action_space, env.action_space)

def run_episode():
    routes=dict()
    episode_rewards=0
    env.reset_reached_points()

    li = list(np.arange(len(carriers)))
    key = np.random.choice(li)
    while key>=0:
        stops, reward = run_task(env,agent,key)
        routes[key]=stops
        episode_rewards+=reward

        if len(li) > 1:
            # print(key, li)
            li.remove(key)
            key = np.random.choice(li)
        else:
            key = -1

    # print('Routes',routes)
    # print('Episode Rewards:',episode_rewards)
    # print('Done!')

    return env,agent,episode_rewards,routes


# run_episode()


def run_n_episodes(env, agent, name="training.gif", n_episodes=1000, render_each=100, fps=1):
    # Store the rewards
    rewards = []
    all_routes=[]
    imgs = []
    all_check=[]
    all_done=[]

    # Experience replay
    for i in tqdm_notebook(range(n_episodes)):

        # Run the episode
        env, agent, episode_reward, routes = run_episode()
        rewards.append(episode_reward)
        all_routes.append(routes)
        all_check.append(env.reached_points)

        if (i+1) % render_each == 0:
            img = env.render(return_img=True)
            imgs.append(img)
            # print(i)
            # print('Routes', routes)
            # print('Episode Rewards:', episode_reward)
            #
            # print(agent.Q)
        print('\n***************\t' + str(i) + '\t******************')

    # print(rewards)
    # print('\n\n**********\n\nAll Routes',all_routes)
    print('\n\n**********\n\nRoutes',all_routes[rewards.index(max(rewards))])
    print('Rewards',max(rewards))
    print('Check',all_check[rewards.index(max(rewards))])
    print(agent.Q)

    with open(f"qtable.pickle", "wb") as f:
        pickle.dump(agent.Q, f)

    # Show rewards
    plt.figure(figsize=(15, 3))
    plt.title("Rewards over training")
    plt.plot(rewards)
    plt.show()

    # Save imgs as gif
    imageio.mimsave(name, imgs, fps=fps)

    # return env, agent


run_n_episodes(env=env, agent=agent)










# env.reached_points[0]=1
# run_episode(env,agent,1)

# [[3.69762835 7.57409778]  0
#  [9.03261303 0.22846886]  1
#  [9.53524493 0.36075049]  2
#  [3.53768079 5.87194668]  3
#  [0.93675762 0.17497906]  4
#  [3.80095545 5.52088867]  5
#  [7.63991365 2.36346141]  6
#  [2.75104152 4.9312129 ]  7
#  [1.60984865 7.91871507]  8
#  [9.22365935 2.32576777]  9
#  [6.85037247 5.61773748]] 10
