import numpy as np
from scipy.spatial.distance import cdist, euclidean
import matplotlib.pyplot as plt


class DeliveryEnvironment:

    def __init__(self, n_carriers, n_stores, n_customers, max_radius=10):
        self.n_carriers = np.array(n_carriers)
        self.n_stores = np.array(n_stores)
        self.n_customers = n_customers
        # print(self.n_customers)
        self.max_radius = max_radius
        self.stops = []
        self.customers = []

        self.max_custs = dict()
        self.curr_custs = dict()

        # print(self.n_customers, '\n')
        self.get_positions()

        self.action_space = len(self.n_carriers) + len(self.n_stores) + len(self.customers)
        print('action_space: ', self.action_space)

        self.generate_q_values()
        # self.render()
        # for first_stop in self.n_carriers:
        #     self.reset(first_stop)

    def get_positions(self):
        # self.carriers = np.random.rand(self.n_carriers,2)*self.max_radius
        # # print(carriers)
        # self.stores = np.random.rand(self.n_stores,2)*self.max_radius
        # self.customers = np.random.rand(self.n_customers,2)*self.max_radius
        self.customers.extend(self.n_customers[0])
        self.customers.extend(self.n_customers[1])
        self.customers.extend(self.n_customers[2])
        self.customers = np.array(self.customers)
        # print(self.customers)

        xy = list()
        xy.extend(self.n_carriers)
        xy.extend(self.n_stores)
        xy.extend(self.customers)

        xy = np.array(xy)

        self.x = xy[:, 0]
        self.y = xy[:, 1]
        # print(xy[:,0])

    def generate_q_values(self):
        xy = np.column_stack([self.x, self.y])
        self.q_stops = cdist(xy, xy)
        # print(self.q_stops)

    def render(self, return_img=False):
        colors = ['g', 'r', 'c', 'm', 'y', 'k', 'w']

        fig = plt.figure(figsize=(7, 7))
        fig = fig.add_subplot(111)
        fig.set_title('Delivery stops')

        fig.scatter(self.n_carriers[:, 0], self.n_carriers[:, 1], c='b', s=50)
        for i in range(len(self.n_stores)):
            fig.scatter(self.n_stores[i, 0], self.n_stores[i, 1], c=colors[i], s=50, marker='P')
            fig.scatter(np.array(self.n_customers[i])[:, 0], np.array(self.n_customers[i])[:, 1], c=colors[i], s=50, marker='*')

        print(self.stops)
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
            fig.canvas.draw_idle()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image

        plt.show()

    def reset(self):
        self.stops = []
        first_stop = np.random.randint(len(self.n_carriers))
        self.stops.append(first_stop)
        return first_stop

    def step(self, destination):
        state = self._get_state()
        new_state = destination

        # reward = 0
        # if self.next_check(state, new_state):
        #     reward = self._get_reward(state, new_state)
        reward = self._get_reward(state, new_state)

        self.stops.append(destination)

        # done = False
        # for x in self.stops:
        #     if x in self.n_stores:
        done = self.customer_to_store_check()
        print('done',done)
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
        print(self.curr_custs,self.max_custs)
        for key in list(self.max_custs.keys()):
            try:
                if self.curr_custs[key] == self.max_custs[key]:
                    flag = flag and True
                else:
                    flag = flag and False
            except:
                flag = flag and False

        return flag

    def set_max_custs(self,new_state):
        if new_state not in list(self.max_custs.keys()):
            self.max_custs[new_state] = np.random.randint(1, len(self.n_customers[new_state - len(self.n_carriers)]) + 1)
        print(self.max_custs)

    def next_check(self, state, new_state):
        # carrier, store
        if state < len(self.n_carriers) <= new_state < len(self.n_stores) + len(self.n_carriers):
            # self.set_max_custs(new_state)
            return True

        # store, store
        elif len(self.n_carriers) <= state < len(self.n_stores) + len(self.n_carriers) and len(
                self.n_carriers) <= new_state < len(
                self.n_stores) + len(self.n_carriers):
            # self.set_max_custs(new_state)
            return True

        # store/customer,customer
        elif new_state >= len(self.n_stores) + len(self.n_carriers) and state >= len(self.n_carriers):
            ordered_store = self.get_store_of_customer([self.x[new_state], self.y[new_state]]) + len(self.n_carriers)
            # print(ordered_store, self.stops)
            if ordered_store in self.stops:
                return True
            else:
                return False

        # customer,store
        elif state >= len(self.n_stores) + len(self.n_carriers) > new_state >= len(self.n_carriers):
            # self.set_max_custs(new_state)
            return True

        # all other
        else:
            return False

    def get_store_of_customer(self, customer):
        key = -1
        for k in list(self.n_customers.keys()):
            vals = self.n_customers[k]
            # print(customer,vals)
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
        q[self.states_memory] = -np.inf

        # print('a',s, [x for x in range(self.actions_size) if x not in self.states_memory and self.env.next_check(s, x)])
        # print('b',s, [x for x in range(self.actions_size) if x not in self.states_memory ])
        # print('c',s, [self.env.next_check(s, x) for x in range(self.actions_size)])

        if np.random.rand() > self.epsilon:
            a = np.argmax(q)
        else:
            a = np.random.choice(
                [x for x in range(self.actions_size) if x not in self.states_memory and self.env.next_check(s, x)])
            # print(a)
            print('dct', env.max_custs)

            # print('a', [x for x in range(self.actions_size) if x not in self.states_memory and self.env.next_check(s, x)])

        return a

    def remember_state(self, s):
        self.states_memory.append(s)
        print('mem', self.states_memory)

    def reset_memory(self):
        self.states_memory = []





def run_episode(env, agent, verbose=1):
    s = env.reset()
    agent.reset_memory()

    max_step = len(env.n_stores) * len(list(env.n_customers.values()))

    episode_reward = 0

    i = 0
    while True:

        # Remember the states
        agent.remember_state(s)

        # Choose an action
        a = agent.action(s)

        # Take the action, and get the reward from environment
        if a >= len(env.n_carriers)+len(env.n_stores):
            k = env.get_store_of_customer([env.x[a], env.y[a]])+len(env.n_carriers)
            # if k not in list(env.curr_custs.keys()):
            #     env.curr_custs[k] = 0
            if env.curr_custs[k] < env.max_custs[k]:
                env.curr_custs[k] += 1
                s_next, r, done = env.step(a)

        elif len(env.n_carriers) <= a < len(env.n_carriers)+len(env.n_stores):
            if (a-len(env.n_carriers)) < len(env.n_stores):
                env.set_max_custs(a)
                env.curr_custs[a] = 0
            s_next, r, done = env.step(a)
        else:
            s_next, r, done = env.step(a)

        # Tweak the reward
        r = -1 * r

        if verbose: print(s_next, r, done)

        # Update our knowledge in the Q-table
        agent.train(s, a, r, s_next)

        # Update the caches
        episode_reward += r
        s = s_next
        print('yet stops',env.stops)
        # If the episode is terminated
        i += 1
        if done:
            print('\n\n\nStops:\n',env.stops)
            break

    return env, agent, episode_reward







carriers = [[3.69762835, 7.57409778], [9.03261303, 0.22846886]]
stores = [[9.53524493, 0.36075049], [3.53768079, 5.87194668], [0.93675762, 0.17497906]]
customers = {0: [[3.80095545, 5.52088867], [7.63991365, 2.36346141],[9.93261303, 0.12846886],[3.53768079, 5.87194668]],
             1: [[2.75104152, 4.9312129], [1.60984865, 7.91871507]],
             2: [[9.22365935, 2.32576777], [6.85037247, 5.61773748],[6.69762835, 3.57409778]]}
# print(carriers)
env = DeliveryEnvironment(carriers, stores, customers)
agent = DeliveryQAgent(env, env.action_space, env.action_space)

run_episode(env, agent)

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
