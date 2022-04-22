
import numpy as np
import matplotlib.pyplot as plt
import copy

from es import *

import RL_formulation as rl
import agent_arch as arch


def apply_solution(solution_not_norm, show=False):

    solution = solution_not_norm / 1 # np.linalg.norm(solution_not_norm)

    num_of_steps = 24
    env = rl.Environment()

    agent = arch.Agent()

    agent.updateParams(solution)

    [[states, actions, total_reward],
     [desired_total_Production_in_dinamics,
      total_Production_in_dinamics,
      run_out_of_elecs]]                 = rl.generate_episode(env, agent, num_of_steps, simple_action=False, show=show)

    num_of_switches = []
    # total_run_out_for_elecs = []
    for j in range(env.num_of_elecs):
        # total_run_out_for_elecs.append(env.Plant[j].total_run_out)
        num_of_switches.append(env.Plant[j].switch_num)

    Production_Error_in_dinamics = np.array(desired_total_Production_in_dinamics) - np.array(total_Production_in_dinamics)

    return [Production_Error_in_dinamics, run_out_of_elecs,
            np.array(total_Production_in_dinamics), np.array(desired_total_Production_in_dinamics), num_of_switches]

def get_num_params():
    agent = arch.Agent()
    param_count = 0
    for param in agent.model.parameters():
      #print(param.data.shape)
      param_count += np.product(param.data.shape)

    del agent

    return param_count

import pickle

def init_train_attributes(from_file = True): # deserialization

    folder = 'serialised_data_CMAES_MarkovDelta_t'

    param_count = get_num_params()

    sigma_init = 0.5  # 0.1

    popsize = 100

    es = CMAES(num_params=param_count, sigma_init=sigma_init, popsize=popsize)

    if from_file:
        file_es = open(folder + '/evalution_strategy.pkl', 'rb')
        old_es = pickle.load(file_es)
        file_es.close()

        es.es = old_es

        del old_es

        #es.es.sigma = 0.6

        file_params = open(folder + '/best_params.pkl', 'rb')

        file_log = open(folder + '/best_scores_in_generations_log.pkl', 'rb')
        file_dict_score_solution = open(folder + '/dict_score_solution.pkl', 'rb')


        best_params = pickle.load(file_params)

        best_scores_in_generations_log = pickle.load(file_log)
        max_min_score_solution = pickle.load(file_dict_score_solution)


        file_params.close()

        file_log.close()
        file_dict_score_solution.close()

        best_score_solution = max_min_score_solution[0]
        worst_score_solution = max_min_score_solution[1]

    else:
        #es.es.x0 = copy.deepcopy(best_params)

        best_params = [0]*param_count

        best_scores_in_generations_log = []

        best_score_solution = [-np.inf, []] # [best_score, best_solution]
        worst_score_solution = [0, []]

    return [es, best_params, best_scores_in_generations_log, best_score_solution, worst_score_solution]


def solution_analysis(solution, n_test_episodes = 6):

    reward_account = rl.Reward()

    score = 0

    for _ in range(n_test_episodes):

        [Production_Error_in_dinamics, total_run_out_for_elecs,
        total_Production_in_dinamics,
        desired_total_Production_in_dinamics,
        num_of_switches]                                            = apply_solution(solution)

        [neg_J, MSE, mean_error, asymetric_error,
        max_total_run_out,
        run_out_deviation_RMSE] = reward_account.account(Production_Error_in_dinamics, total_run_out_for_elecs)

        score += neg_J
    
    sol_J = 0
    for i in range(len(solution)):
        sol_J += int(abs(solution[i]) > 30)
    
    return score / n_test_episodes -0.01*sol_J


from joblib import Parallel, delayed


def train_for_some_generations(generations_num, train_attributes, parallel=True, n_jobs=-1):

    [es, best_params, best_scores_in_generations_log, best_score_solution, worst_score_solution] = train_attributes

    for i in range(generations_num):

        solutions = es.ask()

        if parallel:
            reward_list = Parallel(n_jobs=n_jobs)(delayed(solution_analysis)(solution) for solution in solutions)

        else:
            reward_list = []

            solut_number = 0
            for solution in solutions:  # можно параллельно

                score = solution_analysis(solution)

                reward_list.append(score)

                print(str(solut_number) + ' solution is aplied, score = ' + str(score) + '  min-max = ' + str(
                    min(solution)) + ' -- ' + str(max(solution)))

                if score > best_score_solution[0]:
                    best_score_solution[0] = copy.deepcopy(score)
                    best_score_solution[1] = copy.deepcopy(solution)

                if score < worst_score_solution[0]:
                    worst_score_solution[0] = copy.deepcopy(score)
                    worst_score_solution[1] = copy.deepcopy(solution)

                solut_number += 1

        es.tell(reward_list)

        es_solution = es.result()

#         model_params = es_solution[0] # best historical solution

        best_reward = es_solution[1] # best reward
        curr_best_reward = es_solution[2] # best of the current batch

        curr_best_reward_my_for_validation = max(reward_list)

        print(str(i) + "  ==>>", end=' ')
        print(curr_best_reward_my_for_validation, end=' === ')
        print(curr_best_reward, end=' === ')
        print(best_reward, end=' === ')
        print(es.rms_stdev())


        best_scores_in_generations_log.append(curr_best_reward_my_for_validation)


    return [es.result(), es.current_param()] # best historical solution

def serialization(es, best_solution, best_scores_in_generations_log, best_score_solution, worst_score_solution, folder='serialised_data_CMAES_MarkovDelta_t'):

    file_es = open(folder+'/evalution_strategy.pkl', 'wb')

    pickle.dump(es.es, file_es)

    file_es.close()

    file_params = open(folder+'/best_params.pkl', 'wb')

    file_log = open(folder+'/best_scores_in_generations_log.pkl', 'wb')
    file_dict_score_solution = open(folder+'/dict_score_solution.pkl', 'wb')

    pickle.dump(best_solution, file_params)

    pickle.dump(best_scores_in_generations_log, file_log)

    max_min_score_solution = [best_score_solution, worst_score_solution]
    pickle.dump(max_min_score_solution, file_dict_score_solution)

    file_params.close()

    file_log.close()
    file_dict_score_solution.close()


import sys

#def train_and_seri(generations_num):
if __name__ == "__main__":

    #generations_num = 5

    from_file = True
    parallel = True
    n_jobs = -1

    assert len(sys.argv) > 1, 'Specify the number of generations.'
    
    generations_num = int(sys.argv[1])
    
    if len(sys.argv) > 2:
        assert len(sys.argv) == 5, 'Specify all arguments.'
    
        assert sys.argv[2] == 'not_f' or sys.argv[2] == 'f', 'Specify from_file correctly. (not_f or f)'
    
        if sys.argv[2] == 'not_f':
            from_file = False
        elif sys.argv[2] == 'f':
            from_file = True
        else:
            print(' from_file assert do not work')
    
        assert sys.argv[3] == 'not_paral' or sys.argv[3] == 'paral', 'Specify parallel correctly. (not_paral or paral)'
    
        if sys.argv[3] == 'not_paral':
             parallel = False
        elif sys.argv[3] == 'paral':
            parallel = True
        else:
            print('parallel assert do not work')
    
        n_jobs = int(sys.argv[4])

    [es, best_params, best_scores_in_generations_log, best_score_solution, worst_score_solution] = init_train_attributes(from_file=from_file)

    print('start training')

    train_Report = train_for_some_generations(generations_num,
                                              [es, best_params, best_scores_in_generations_log, best_score_solution, worst_score_solution],
                                              parallel=parallel,
                                              n_jobs=n_jobs)

    print('end training')
    print()

    curr_solution = train_Report[1]

    best_solution = train_Report[0][0]

    print('max of curr_solution = ' + str(max(curr_solution)))
    print('min of curr_solution = ' + str(min(curr_solution)))
    print()

    print('length of best_scores_in_generations_log = ' + str(len(best_scores_in_generations_log)))

    best_score_solution = [train_Report[0][1], best_solution]

    print('best_score_solution = ')
    print(best_score_solution)
    print()

    print('curr_solution = ')
    print(curr_solution)
    print()

    # simulation and score account
    [Production_Error_in_dinamics, total_run_out_for_elecs,
     total_Production_in_dinamics, desired_total_Production_in_dinamics, num_of_switches] = apply_solution(
        curr_solution)

    reward_account = rl.Reward()

    [neg_J, MSE, mean_error, asymetric_error,
     max_total_run_out,
     run_out_deviation_RMSE] = reward_account.account(Production_Error_in_dinamics, total_run_out_for_elecs)

    del reward_account

    print('=========================== SCORE of the current soluion')
    print('neg_J = ' + str(neg_J))
    print('RMSE = ' + str(MSE ** 0.5))
    print('mean_error = ' + str(mean_error))
    print('asymetric_error = ' + str(asymetric_error))
    print('max_total_run_out = ' + str(max_total_run_out))
    # print(min_max_tot_run_out)
    print('run_out_deviation_RMSE = ' + str(run_out_deviation_RMSE))
    print("switc num = ", end=' ')
    print(num_of_switches)
    print(sum(num_of_switches))
    print(max(total_run_out_for_elecs) / sum(total_run_out_for_elecs))
    print()


    [Production_Error_in_dinamics, total_run_out_for_elecs,
     total_Production_in_dinamics, desired_total_Production_in_dinamics, num_of_switches] = apply_solution(
        best_solution)

    reward_account = rl.Reward()

    [neg_J, MSE, mean_error, asymetric_error,
     max_total_run_out,
     run_out_deviation_RMSE] = reward_account.account(Production_Error_in_dinamics, total_run_out_for_elecs)

    del reward_account

    print('=========================== SCORE of the best solution')
    print('neg_J = ' + str(neg_J))
    print('RMSE = ' + str(MSE ** 0.5))
    print('mean_error = ' + str(mean_error))
    print('asymetric_error = ' + str(asymetric_error))
    print('max_total_run_out = ' + str(max_total_run_out))
    # print(min_max_tot_run_out)
    print('run_out_deviation_RMSE = ' + str(run_out_deviation_RMSE))
    print("switc num = ", end=' ')
    print(num_of_switches)
    print(sum(num_of_switches))
    print(max(total_run_out_for_elecs) / sum(total_run_out_for_elecs))
    print()

    '''
    # plot PR and PR_ref
    plt.figure(figsize=(30, 15))
    plt.title("Electrolyser modeling")
    plt.plot(desired_total_Production_in_dinamics, label='I_ref')
    plt.plot(total_Production_in_dinamics, label='I')

    plt.legend()
    plt.grid(visible=True)
    plt.show()


    # plot hist
    num_elecs = 5
    names = list(range(num_elecs))
    values = total_run_out_for_elecs

    plt.figure(figsize=(30, 15))
    plt.bar(names, values)
    plt.show()


    # plot neg_loss
    plt.figure(figsize=(30, 15))
    plt.title("log_generation")
    plt.plot(best_scores_in_generations_log, label='neg_loss')

    plt.legend()
    plt.grid(visible=True)
    plt.show()
    ''' 
    # serialization
    serialization(es, best_solution, best_scores_in_generations_log, best_score_solution, worst_score_solution,
                  folder='serialised_data_CMAES_MarkovDelta_t')






