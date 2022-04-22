
'''
На каждом шаге марковского процесса, (каждые Delta_t секунд),
среда выдает новое состояние и награду в соответствии с тем, какое было совершено действие агентом.
Агент получает новое состояние и выбирает следующее действие, которое передается в среду и ее состояние меняется.
'''

import numpy as np
import copy
import torch
import math

import formatting as frmt
# import network_arch as arch

class Environment:
    def __init__(self):
        self.Plant = []
        self.num_of_elecs = 5
        self.init_state_of_electrolysers = [['idle', 30.0, 0, 0] for _ in range(self.num_of_elecs)]

        self.delta_t = 1
        self.Delta_t = 15 * 60

        self.PRref = []
        self.PRref_noised = [] # это кривая желаемой суммарной выработки на следующие self.num_of_steps_left шагов
        # каждый шаг это self.Delta_t секунд. self.delta_t это шаг дискретизации.

        self.reward_account = Reward()

        self.num_of_steps_left = 24 # количество 15минуток в 6 часах

    def reset(self, num_of_steps_left):
        self.num_of_steps_left = num_of_steps_left

        [self.PRref, self.PRref_noised] = frmt.get_PRref_init()
        self.Plant = frmt.init_Plant(self.num_of_elecs, self.init_state_of_electrolysers, self.delta_t)
        init_state = frmt.formate_state(self.Plant, self.PRref_noised / (self.num_of_elecs * 100))
        # делим на максимальную суммарную выработку со всех электролизеров электролизеров
        # чтобы получить относительное значение желаемого суммарного уровня выработки
        return init_state
    
    def interpret_action(self, action, simple_action): # simple_action означает, что 
        # action это num_of_elecs мерный вектор с вещественными значениями из отрезка [-1, 1]
        # иначе action это 2*num_of_elecs мерный вектор с вещественными значениями из отрезка [-1, 1]
        
        # -1 <= action[j] <= 1.
        assert all([-1 <= item <= 1 for item in action])

        if simple_action:
            # action это num_of_elecs мерный вектор с вещественными значениями из отрезка [-1, 1]

            ne = self.num_of_elecs

            U = []

            for j in range(self.num_of_elecs):
                if action[j] < -0.9:
                    U.append(0)
                else:
                    U.append( 40*(action[j])/1.9 + 100-(40/1.9) )

            return U

        else:
            # action это 2*num_of_elecs мерный вектор с вещественными значениями из отрезка [-1, 1]
            ne = self.num_of_elecs

            U = []

            for j in range(self.num_of_elecs):
                if action[j] < 0:
                    U.append(0)
                else:
                    U.append(60 + 40 * ((action[j + ne] + 1) / 2))  # = 40*(action[j])/1.9 + 100-(40/1.9)

            return U

    def step_continuous(self, action, simple_action=False, made=False): # made означает, что action
        # не нужно предобрабатывать и можно его интерпретировать, как U
        
        if made:
            U = copy.deepcopy(action)
        else:
            U = self.interpret_action(action, simple_action)

        total_Production_in_dinamics = []
        assert self.PRref[0] == self.PRref_noised[0]
        desired_total_Production_in_dinamics = self.PRref[0] * np.ones(int(self.Delta_t / self.delta_t))

        run_out_of_elecs_before = [0] * self.num_of_elecs
        for j in range(self.num_of_elecs):
            run_out_of_elecs_before[j] = self.Plant[j].total_run_out

        for i in range(int(self.Delta_t / self.delta_t)):
            production_from_elecs_in_moment = 0
            for j in range(self.num_of_elecs):
                elec = self.Plant[j]

                elec.apply_control_signal_in_moment(U[j], i * self.delta_t) # не правильно определено t = i * self.delta_t время не должно обнуляться на каждой 15минутке, но сейчас это не имеет значения, потому что время используется только для хитинга, а во время хитинга управление не происходит
                
                [y, yd, ydd] = elec.getDinamics()
                #[Temper, Temper_d] = elec.getTemperatureDinamics()

                production_from_elecs_in_moment += y

            total_Production_in_dinamics.append(production_from_elecs_in_moment*100)

        [self.PRref, self.PRref_noised] = frmt.Generator_of_desired_PR(self.delta_t, self.Delta_t, self.PRref, self.PRref_noised)

        new_state = frmt.formate_state(self.Plant, self.PRref_noised / (self.num_of_elecs * 100))
        # делим на максимальную суммарную выработку со всех электролизеров (self.num_of_elecs * 100)
        # чтобы получить относительное значение желаемого суммарного уровня выработки

        run_out_of_elecs_after = [0] * self.num_of_elecs
        for j in range(self.num_of_elecs):
            run_out_of_elecs_after[j] = self.Plant[j].total_run_out

        run_out_of_elecs = np.array(run_out_of_elecs_after) - np.array(run_out_of_elecs_before)

        Production_Error_in_dinamics = np.array(desired_total_Production_in_dinamics) - np.array(total_Production_in_dinamics)

        self.num_of_steps_left -= 1

        done = (self.num_of_steps_left == 0)

        info = [desired_total_Production_in_dinamics, total_Production_in_dinamics]

        # print("self.num_of_steps_left")
        # print(self.num_of_steps_left)
        # print()

        return [new_state, Production_Error_in_dinamics, run_out_of_elecs, done, info]

    def step_discrete(self, action):
        # action это номер дискретного действия. всего действий 2*self.num_of_elecs + 1.
        # первые self.num_of_elecs действий соответствуют номеру электролизера, который нужно включить,
        # следующие self.num_of_elecs действий соответствуют номеру электролизера, который нужно выключить,
        # последнее действие означает ничего не делать

        # мы видим профиль self.PRref_noised, но какое решение принять для self.PRref_noised[0], чтобы оно оказалось оптимальным
        # на всем self.PRref_noised? если менять уровень выработки, то это не влияет на износ, но можно уменьшить для
        # self.PRref_noised[0] ско, но возможно, если сейчас сделать действие х1 (например, изменить уровень выработки),
        # которое в данный момент оптимально это повлияет на то что в дальнейшем придется сделать действие у1.
        # а если сделать действие х2, которое не оптимально сейчас, но позволит дальше сделать действие у2. х2, у2 могут в
        # совокупности оказаться более оптимальной стратегией, чем х1, у1.

        #self.PRref_noised

        pass

        #return [new_state, Production_Error_in_dinamics, run_out_of_elecs, done, info]

# class Agent:
#     def __init__(self):
#         self.model = arch.DNetwork()

#         orig_model = copy.deepcopy(self.model)

#         self.model_shapes = []
#         for param in orig_model.parameters():
#             p = param.data.cpu().numpy()
#             self.model_shapes.append(p.shape)

#     def updateParams(self, flat_param: np.array):
#         idx = 0
#         i = 0

#         for param in self.model.parameters():
#             delta = np.product(self.model_shapes[i])
#             block = flat_param[idx:idx + delta]
#             block = np.reshape(block, self.model_shapes[i])
#             i += 1
#             idx += delta
#             block_data = torch.from_numpy(block).float()

#             param.data = block_data


#     def getAction(self, state):

#         num_elecs = 5

#         assert all([abs(item) <= 1.1 for item in state])

#         act_logist_regression = self.model(torch.from_numpy(state).float()).detach().numpy()

#         action = []

#         for j in range(num_elecs):
#             if act_logist_regression[j] == 0:
#                 action.append(0.0)
#             else:
#                 action.append(act_logist_regression[j + num_elecs])

#         assert all([abs(item) <= 1 for item in action])

#         return np.array(action) # action[i] \in {0} \cup [0.6, 1]

#     def getActionEmpirical(self, curve_of_desired_total_output, Plant):
#         # curve_of_desired_total_current[k] is a reference total OUTPUT at time [k*Delta_t . . . (k+1)Delta_t)

#         desired_output_at_next_Delta_t = curve_of_desired_total_output[0]

#         Total_output = 0  # в долях от максимальной выработки одного электролизера

#         working_elecs = []
#         not_working_elecs = []

#         U = [0] * len(Plant)

#         for i in range(len(Plant)):
#             elec = Plant[i]
#             if elec.getCurrentTarget() != 0:
#                 working_elecs.append(elec)
#             else:
#                 not_working_elecs.append(elec)

#             U[i] = elec.getCurrentTarget() * 100

#             Total_output += Plant[i].getCurrentTarget()

#         number_of_required_new_electrolysers = 0
#         newTotal_output = Total_output
#         while abs(newTotal_output - desired_output_at_next_Delta_t) > 1 / 2:
#             # ошибаемся больше чем на половину выработки одного электролизера
#             if newTotal_output < desired_output_at_next_Delta_t:
#                 number_of_required_new_electrolysers += 1
#                 newTotal_output += 1 # включаем + один электролизер на 100%
#             else:  # newTotal_output > desired_Rate_at_next_Delta_t
#                 number_of_required_new_electrolysers -= 1
#                 newTotal_output -= 1 # включаем один электролизер полностью

#         if number_of_required_new_electrolysers < 0:  # нужно выключить несколько
#             # выключаем те которые самые горячие
#             # выключаем те которые самые изношенные *
#             # выключаем те которые дольше всех работают

#             working_elecs = sorted(working_elecs, key=lambda x: -x.total_run_out)
#             num_of_elecs_to_off = -number_of_required_new_electrolysers
#             for i in range(min(num_of_elecs_to_off, len(working_elecs))):
#                 U[working_elecs[i].getID()] = 0

#         elif number_of_required_new_electrolysers > 0:  # нужно включить несколько
#             # включаем те которые дольше всех выключены
#             # самые холодные *
#             # самые не изношенные

#             not_working_elecs = sorted(not_working_elecs, key=lambda x: x.getTemperatureDinamics()[0])
#             for i in range(min(number_of_required_new_electrolysers, len(not_working_elecs))):
#                 U[not_working_elecs[i].getID()] = 100

#         return np.array(U) / 100  # U[i] \in {0%} \cup [60%,100%]

#     def fit_policy(self, elite_trajectory):
#         pass

class Reward:
    def __init__(self):
        self.loss = "cross entropy"

    def account(self, Production_Error_in_dinamics, run_out_of_elecs):

        MSE = (np.dot(Production_Error_in_dinamics,
                       Production_Error_in_dinamics) / Production_Error_in_dinamics.size) # ** 0.5  # !

        mean_error = Production_Error_in_dinamics.mean()

        asymetric_error = frmt.asymetric_plus(mean_error, 2, 1)  # !

        max_total_run_out = run_out_of_elecs.max() # !
        mean_total_run_out = run_out_of_elecs.mean()

        deviation = run_out_of_elecs - (mean_total_run_out * np.ones_like(run_out_of_elecs))

        run_out_deviation_RMSE = (np.dot(deviation, deviation) / deviation.size) ** 0.5  # !


        log_J = self.get_score(MSE, mean_error, asymetric_error, max_total_run_out, run_out_deviation_RMSE)

        return [-log_J, MSE, mean_error, asymetric_error, max_total_run_out, run_out_deviation_RMSE]
    
    def get_score(self, MSE, mean_error, asymetric_error, max_total_run_out, run_out_deviation_RMSE):
        
        assert not any([math.isinf(abs(item)) for item in [MSE, mean_error, asymetric_error, max_total_run_out, run_out_deviation_RMSE]])
        
        gamma1 = 0.3
        #gamma2 = 0.002
        gamma3 = 0.00002  # 0.00002
        gamma5 = 0.0002

        log_J = gamma1 * MSE**0.5 + gamma3 * max_total_run_out + gamma5 * run_out_deviation_RMSE # gamma2 * asymetric_error
        
        return log_J
        


def generate_episode(env, agent, num_of_steps, show=False, simple_action=False, made=False, manual=False):
    desired_total_Production_in_dinamics = []
    total_Production_in_dinamics = []
    
    run_out_of_elecs = np.zeros(5)

    states, actions = [], []
    total_reward = 0

    state = env.reset(num_of_steps)

    for k in range(num_of_steps):
        
        if show:

            print("PRref")
            print(env.PRref_noised)
            print()

            print("state")
            print(state)
            print()

        if made:
            
            if manual:
                action = agent.getAction_Manual()
            
            else:            
                # action = agent.getActionEmpirical(env.PRref_noised/100, env.Plant)
                action = agent.getActionEmpirical_with_PR_control(env.PRref_noised/100, env.Plant)
        else:
            action = agent.getAction(state)
        
        if show:
            print("action")
            print(action)
            print()
            
            if not made:
                OUT = env.interpret_action(action, simple_action)

                print("OUT")
                print(OUT)
                print()

                print("last laier")
                print(agent.last_laier)
                print()

                print("last laier before saturation")
                print(agent.model.last_laier_befor_saturation)
                print()


        [new_state, Production_Error_in_dinamics, run_out_of_elecs_on_step, done, info] = env.step_continuous(action, simple_action=simple_action, made=made)

#         [neg_J, MSE, mean_error, asymetric_error,
#          max_total_run_out,
#          run_out_deviation_RMSE] = env.reward_account.account(Production_Error_in_dinamics, run_out_of_elecs)

#         if show:
#             print("reward")
#             print(neg_J)
#             print()

        states.append(state)
        actions.append(action)
#         total_reward += neg_J  # neg_J is a reward

        desired_total_Production_in_dinamics.extend(info[0])
        total_Production_in_dinamics.extend(info[1])
        
        
        run_out_of_elecs += run_out_of_elecs_on_step

        state = new_state
        
        if done:
            if show:
                print("DONE")
                
    [neg_log_J, MSE, mean_error, asymetric_error,
         max_total_run_out,
         run_out_deviation_RMSE] = env.reward_account.account(Production_Error_in_dinamics, run_out_of_elecs)
    
    total_reward = neg_log_J

    return [[states, actions, total_reward], [desired_total_Production_in_dinamics,
                                              total_Production_in_dinamics,
                                              run_out_of_elecs]]





