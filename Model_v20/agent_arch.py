

import numpy as np
import copy

import torch
import torch.nn as nn

import torch.optim as optim

from scipy.stats import multivariate_normal

from es import CMAES, PEPG

import RL_formulation as rl

import matplotlib.pyplot as plt

from joblib import Parallel, delayed

class DNetwork(nn.Module):
    def __init__(self):
        super(DNetwork, self).__init__()

        self.state_dim = 99
        self.action_dim = 10

        hiden_layer_len_1 = 50
        hiden_layer_len_2 = 50

        self.hidden_1 = nn.Linear(self.state_dim, hiden_layer_len_1, bias=True)

        self.hidden_2 = nn.Linear(hiden_layer_len_1, hiden_layer_len_2, bias=True)

        self.output = nn.Linear(hiden_layer_len_2, self.action_dim, bias=True)


        self.activation_on_hidden_1 =  nn.ReLU() # nn.Tanh() # nn.ReLU()

        #self.activation_on_hidden_2 = nn.Tanh() #

        self.activation_on_output = nn.Tanh()  # nn.Softmax()

        self.last_laier_befor_saturation = []


    def forward(self, x):

        x = self.hidden_1(x)
        x = self.activation_on_hidden_1(x)
        x = self.hidden_2(x)
        #x = self.activation_on_hidden_2(x)
        x = self.output(x)
        #x = self.activation_on_output(x)

        # ======================================= !

        #x = self.Norm(x, self.sigma*torch.eye(2)).sample()

        self.last_laier_befor_saturation = copy.deepcopy(x.detach().numpy())

        x = self.activation_on_output(x) # torch.maximum(torch.minimum(x, torch.tensor(1)), torch.tensor(-1)) # self.activation_on_output(x) #

        return x



class Agent:

    def __init__(self):
        self.model = DNetwork()

        self.model_shapes = []

        orig_model = copy.deepcopy(self.model)

        for param in orig_model.parameters():
            p = param.data.cpu().numpy()
            self.model_shapes.append(p.shape)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=0.002)


#         self.Norm = torch.distributions.multivariate_normal.MultivariateNormal

#         self.sigma = torch.tensor(0.1)

        self.sigma = 0.001

        self.last_laier = []


    def updateParams(self, flat_param: np.array):
        idx = 0
        i = 0

        for param in self.model.parameters():
            delta = np.product(self.model_shapes[i])
            block = flat_param[idx:idx + delta]
            block = np.reshape(block, self.model_shapes[i])
            i += 1
            idx += delta
            block_data = torch.from_numpy(block).float()

            param.data = block_data


    def getAction(self, state):

        with torch.no_grad():

            mu = self.model(torch.from_numpy(state).float()) # .detach().numpy()

        self.last_laier = copy.deepcopy(mu.numpy())

        action = multivariate_normal(mean=mu.numpy(), cov=self.sigma*np.eye(self.model.action_dim)).rvs(size=1) # copy.deepcopy(mu.numpy()) #

        np.clip(action, -1, 1, out= action)

        return action

    def fit(self, states, actions):

        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).float()

        #self.model.train()

        self.optimizer.zero_grad()

        guessies = self.model(states)
        loss = self.criterion(guessies, actions)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def getActionEmpirical(self, curve_of_desired_total_output, Plant):
        # curve_of_desired_total_current[k] is a reference total OUTPUT at time [k*Delta_t . . . (k+1)Delta_t)

        desired_output_at_next_Delta_t = curve_of_desired_total_output[0]

        Total_output = 0  # в долях от максимальной выработки одного электролизера

        working_elecs = []
        not_working_elecs = []

        U = [0] * len(Plant)

        for i in range(len(Plant)):
            elec = Plant[i]
            if elec.getCurrentTarget() != 0:
                working_elecs.append(elec)
            else:
                not_working_elecs.append(elec)

            U[i] = elec.getCurrentTarget() * 100

            Total_output += elec.getCurrentTarget()

        number_of_required_new_electrolysers = 0
        newTotal_output = Total_output
        while abs(newTotal_output - desired_output_at_next_Delta_t) > 1 / 2:
            # ошибаемся больше чем на половину выработки одного электролизера
            if newTotal_output < desired_output_at_next_Delta_t:
                number_of_required_new_electrolysers += 1
                newTotal_output += 1 # включаем + один электролизер на 100%
            else:  # newTotal_output > desired_Rate_at_next_Delta_t
                number_of_required_new_electrolysers -= 1
                newTotal_output -= 1 # включаем один электролизер полностью

        if number_of_required_new_electrolysers < 0:  # нужно выключить несколько
            # выключаем те которые самые горячие
            # выключаем те которые самые изношенные *
            # выключаем те которые дольше всех работают

            working_elecs = sorted(working_elecs, key=lambda x: -x.total_run_out)
            num_of_elecs_to_off = -number_of_required_new_electrolysers
            for i in range(min(num_of_elecs_to_off, len(working_elecs))):
                U[working_elecs[i].getID()] = 0

        elif number_of_required_new_electrolysers > 0:  # нужно включить несколько
            # включаем те которые дольше всех выключены
            # самые холодные *
            # самые не изношенные

            not_working_elecs = sorted(not_working_elecs, key=lambda x: x.getTemperatureDinamics()[0])
            for i in range(min(number_of_required_new_electrolysers, len(not_working_elecs))):
                U[not_working_elecs[i].getID()] = 100

        return np.array(U) # U[i] \in {0%} \cup [60%,100%]
    
    
    def getActionEmpirical_with_PR_control(self, curve_of_desired_total_output, Plant):
        # curve_of_desired_total_current[k] is a reference total OUTPUT at time [k*Delta_t . . . (k+1)Delta_t)

        desired_output_at_next_Delta_t = curve_of_desired_total_output[0]

        Total_output = 0  # в долях от максимальной выработки одного электролизера

        working_elecs = []
        not_working_elecs = []

        U_per100 = [0] * len(Plant)

        for i in range(len(Plant)):
            elec = Plant[i]
            if elec.getCurrentTarget() != 0:
                working_elecs.append(elec)
            else:
                not_working_elecs.append(elec)

            U_per100[i] = elec.getCurrentTarget() # * 100

            Total_output += elec.getCurrentTarget()

        eps = 0.5 # или 0.1 но тогда нужно учесть что не возможно вырабатывать меньше чем 60% и больше чем 0%

        w = 0

        while abs(desired_output_at_next_Delta_t - Total_output) > eps:
            
            print("desired_output_at_next_Delta_t = " + str(desired_output_at_next_Delta_t))
            print("Total_output = " + str(Total_output))

            er = (desired_output_at_next_Delta_t - Total_output)

            if er < 0: # вырабатывается больше чем надо
                er_abs = abs(er)
                workers_on_60rate_num = 0 # количество электролизеров, которые уже опущены до 60
                for j2 in range(len(working_elecs)):  # просто последовательно проходим по электролизерам
                    target = U_per100[working_elecs[j2].getID()] # / 100
                    if target > 0.6:
                        U_per100[working_elecs[j2].getID()] -= min(er_abs, target-0.6) # * 100
                        Total_output -= min(er_abs, target-0.6)
                        break
                    else:
                        workers_on_60rate_num += 1

                if workers_on_60rate_num == len(working_elecs): # так совпало, что когда len(working_elecs) == 0, сюда все таки войдем
                    # все уже работают на 60%
                    # вообще то нужно решить, стоит ли кого нибудь выключать
                    # но сейчас просто выключаем какого-то (последнего из списка) если есть кого выключать

                    if len(working_elecs) == 0:
                        print("len(working_elecs) == 0 & er < 0")
                        print(desired_output_at_next_Delta_t)
                        print(Total_output)
                        print()

                    else:

                        working_elec = working_elecs.pop()
                        Total_output -= U_per100[working_elec.getID()] # /100
                        U_per100[working_elec.getID()] = 0
                        not_working_elecs.append(working_elec)

            else: # вырабатывается меньше чем надо
                max_workers_num = 0 # количество электролизеров, которые работают на максимальном уровне
                for j1 in range(len(working_elecs)): # просто последовательно проходим по электролизерам
                    target = U_per100[working_elecs[j1].getID()] # / 100
                    if target < 1:
                        U_per100[working_elecs[j1].getID()] += min(er, 1-target) # * 100
                        Total_output += min(er, 1-target)

                        print("rise to er " + str(j1))
                        break

                    else:
                        max_workers_num += 1

                if max_workers_num == len(working_elecs): # так совпало, что когда len(working_elecs) == 0, сюда все таки войдем
                    # нужно включить новый потому что все работают на 100%
                    # берем просто какой-то неработающий электролизер (пусть последний из списка)

                    # еще нужно решить стоит ли его включать
                    # но сейчас просто включаем какой то, что бы не отставать от задания

                    if len(not_working_elecs) == 0: # некого включать
                        break
                    else:
                        not_working_elec = not_working_elecs.pop()
                        U_per100[not_working_elec.getID()] = 0.6 # 60
                        Total_output += 0.6
                        working_elecs.append(not_working_elec)

                        print("switch on 60")

            w += 1

        print([elec.ID for elec in working_elecs])
        print([elec.ID for elec in not_working_elecs])
        print(U_per100)
        print(w)
        print()

        
        return np.array(U_per100) * 100 # U[i] \in {0%} \cup [60%,100%]

    
    def getAction_OnlineOptimization(self, curve_of_desired_total_PR, Plant, Delta_t, delta_t, parallel = False): # MPC
        # curve_of_desired_total_PR[k] is a reference total PR at time [k*Delta_t . . . (k+1)Delta_t)
        # в долях от максимальной выработки одного электролизера



        n = len(Plant)
        N = len(curve_of_desired_total_PR)

        def format_solution_on_Delta_t(solution_slice):
            U = np.zeros(len(solution_slice))
            np.clip(solution_slice, -1, 1, out=U)

            for j in range(len(U)):
                if U[j] < -0.9:
                    U[j] = 0
                else:
                    U[j] = 40 * (U[j]) / 1.9 + 100 - (40 / 1.9)

            return U

        def loss_for_control_signal(solution):

            U_J = 0
            for x in solution:
                U_J += (abs(x) - 1)

            # for k in range(N):
            #     U_k = format_solution_on_Delta_t(solution[k*n : (k+1)*n])

            return U_J


        def solution_analysis(solution):
            # solution is a vector of shape (len(curve_of_desired_total_output) * len(Plant), 1)

            local_Plant_for_modelling = copy.deepcopy(Plant)

            total_Production_in_dinamics = []
            desired_total_Production_in_dinamics = []

            run_out_of_elecs_before = [elec.total_run_out for elec in local_Plant_for_modelling]

            for k in range(N):
                U_k = format_solution_on_Delta_t(solution[k*n : (k+1)*n])

                desired_total_Production_in_dinamics.extend(curve_of_desired_total_PR[k] * np.ones(int(Delta_t / delta_t)))

                for i in range(int(Delta_t / delta_t)):
                    production_from_elecs_in_moment = 0
                    for j in range(n):
                        elec = local_Plant_for_modelling[j]

                        t = (k * int(Delta_t / delta_t) + i) * delta_t # момент времени используется отлько чтобы засечь время хитинга, но и там можно его использования избежать. в общем это не важный параметр
                        elec.apply_control_signal_in_moment(U_k[j], t)

                        [y, yd, ydd] = elec.getDinamics()
                        # [Temper, Temper_d] = elec.getTemperatureDinamics()

                        production_from_elecs_in_moment += y

                    total_Production_in_dinamics.append(production_from_elecs_in_moment * 100)

            run_out_of_elecs_after = [elec.total_run_out for elec in local_Plant_for_modelling]

            total_run_out_for_elecs = np.array(run_out_of_elecs_after) - np.array(run_out_of_elecs_before)

            Production_Error_in_dinamics = np.array(desired_total_Production_in_dinamics) - np.array(
                total_Production_in_dinamics)

            reward_account = rl.Reward()

            [neg_J, MSE, mean_error, asymetric_error,
             max_total_run_out,
             run_out_deviation_RMSE] = reward_account.account(Production_Error_in_dinamics, total_run_out_for_elecs)

            assert neg_J < 0

            return neg_J -0.2*loss_for_control_signal(solution)

        def prepare_action(training_result):
            [es_result, es_current_param] = training_result

            curr_solution = es_current_param

            best_solution = es_result[0]

            best_solution_score = es_result[1]

            U_for_N_Delta_t = []
            for k in range(N):
                U_k = format_solution_on_Delta_t(best_solution[k*n : (k+1)*n])
                U_for_N_Delta_t.append(U_k)

            return [U_for_N_Delta_t, best_solution_score]


        n_params = N * n # 24 * 5
        n_populations = 20
        sigma_init = 0.5

        n_generations = 100 # fit ===============================

        es = CMAES(num_params=n_params, sigma_init=sigma_init, popsize=n_populations) # PEPG CMAES

        best_scores_in_generations_log = []

        for g in range(n_generations):
            solutions = es.ask()

            if parallel:
                reward_list = Parallel(n_jobs=-1)(delayed(solution_analysis)(solution) for solution in solutions)

            else:
                reward_list = []

                solut_number = 0
                for solution in solutions:  # можно параллельно

                    score = solution_analysis(solution)

                    reward_list.append(score)

                    print(str(solut_number) + ' solution is aplied, score = ' + str(score) + '  min-max = ' + str(
                        min(solution)) + ' -- ' + str(max(solution)))

                    solut_number += 1

            curr_best_reward = max(reward_list)

            es.tell(reward_list)

            # [es.result(), es.current_param()]

            best_scores_in_generations_log.append(curr_best_reward)

            print(str(g) + "  ==>>", end=' ')
            print(curr_best_reward, end=' === ')
            print(es.result()[1], end=' === ')
            print(es.rms_stdev(), end=' === ')
            print('min-max = ' + str(
                        min(es.result()[0])) + ' -- ' + str(max(es.result()[0])))

        # plot neg_loss
        plt.figure(figsize=(30, 15))
        plt.title("log_generation")
        plt.plot(best_scores_in_generations_log, label='neg_loss')

        plt.legend()
        plt.grid(visible=True)
        plt.show()

        [U_for_N_Delta_t, best_solution_score] = prepare_action([es.result(), es.current_param()])

        action = U_for_N_Delta_t[0]

        return [action, U_for_N_Delta_t, best_solution_score]
    
    
    
    def getAction_Manual(self):
        
        # ручное управление
        
        return list(map(float, input().split()))







#
#
# import numpy as np
# import torch
# import torch.nn as nn
#
# def linear_function(x, k):
#     return k*x
#
# class DNetwork(nn.Module):
#     def __init__(self):
#         super(DNetwork, self).__init__()
#
#         state_len = 99
#         num_elecs = 5
#
#         hiden_layer_len_1 = 50
#         hiden_layer_len_2 = 50
#
#
#         self.hidden_1 = nn.Linear(state_len, hiden_layer_len_1, bias=True)
#         self.hidden_2 = nn.Linear(hiden_layer_len_1, hiden_layer_len_2, bias=True)
#
#         self.output = nn.Linear(hiden_layer_len_2, 2 * num_elecs, bias=True)
#
#
#         self.activation_on_hidden_1 = nn.ReLU()
#         self.activation_on_hidden_2 = linear_function
#
#         self.activation_on_output = linear_function
#
#         # для проверки
#         self.OUT = []
#
#     def forward(self, x):
#         x = self.hidden_1(x)
#         x = self.activation_on_hidden_1(x)
#         x = self.hidden_2(x)
#         x = self.activation_on_hidden_2(x, 1)
#         x = self.output(x)
#         x = self.activation_on_output(x, 1)
#
#         n2 = len(x)
#
#         # для проверки
#         self.OUT = []
#         for x_j in x.detach().numpy():
#             self.OUT.append(x_j)
#
#         assert n2 == 10
#
# #         regression_half = torch.narrow(x, dim=0, start= n2//2, length=n2//2)
# #         norm_of_regression_half1 = np.linalg.norm(regression_half.detach().numpy())
#
#         for j in range(n2):
#
#             if j < n2 // 2:
#                 if x[j] <= 0:
#                     x[j] = 0
#                 else:
#                     x[j] = 1
#
#             else:
#                 x[j] = torch.tensor(0.2) * (torch.maximum(torch.minimum(x[j], torch.tensor(1)),
#                                                           torch.tensor(-1)) + torch.tensor(1)) + torch.tensor(0.6)
#
#         # if not all([ (abs(item) <= 1 and item != np.nan) for item in x.detach().numpy()]):
#         #     print('\n' + "================================ THIS ACTION NON ==================================")
#         #     print(x)
#         #     print(self.OUT)
#         #     print("================================ THIS ACTION NON ==================================" + '\n')
#
#         assert all([ (abs(item) <= 1 and item != np.nan) for item in x.detach().numpy()]) # != np.nan
#
#         return x
#
#
