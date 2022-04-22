
import numpy as np
from scipy.stats import norm
from electrolyser import Electrolyser

min_production = 0 # % от максимальной выработки одного электролизера
max_production = 500 # % от максимальной выработки одного электролизера

def get_PRref_init():
    # 24 = 6 hours * (60 / 15) PR_experimental[:24] #
    # np.random.choice(values, size=24)

    Delta_t = 15 * 60  # 15 minutes
    horizon_size = 6 * 60 * 60  # 6 hours

    PRref = np.random.uniform(min_production, max_production, int(horizon_size / Delta_t))
    PRref_noised = PRref.copy()

    # зашумление
    #PRref_noised[0] = PRref[0]
    for k in range(1, len(PRref)):  # первое значение не зашумляется, оно достоверное
        alpha = PRref[k] * 5 / 100
        PRref_k = norm_sample(mu=PRref[k], sigma=alpha * k)
        extra = 0  # можно увеличить колокол, а то граничные значения имеют обрезанные колоколы
        PRref_noised[k] = min(max(PRref_k, min_production + extra), max_production)

    # print("PRref_init")
    # print(PRref)
    # print()
    #
    # print("PRref_noised_init")
    # print(PRref_noised)
    # print()

    return [PRref, PRref_noised] # 510 потому что 5 электролизеров и максимум могут роботать на 500

def norm_sample(mu, sigma):

    return norm(mu, sigma).rvs(size=1)[0]

def Generator_of_desired_PR(delta_t, Delta_t, PRref, PRref_noised):

    # к нему обращаются каждые Delta_t секунд (15 минут) и он возвращает временной ряда на следующие 6 часов
    # с разрешением Delta_t секунд, в котором первое значение, то есть на ближайшие Delta_t секунд достоверное, а на последующие
    # 15минутки -
    # - это прогноз (зашумленные значениея). этот временной ряд каждые 15 минут обновляется (первое значение удаляется,
    # и в конец добавляется новое). Каждые 5 минут ряд поновому зашумляется, как бы приходит новый прогноз.

    # обновление

    PRref = np.delete(PRref, 0)
    last = PRref[-1]
    next_value = last + norm_sample(mu=0, sigma=200) 
    next_value = max(min(next_value, max_production), min_production)
    PRref = np.append(PRref, [next_value])

    # зашумление

    PRref_noised[0] = PRref[0]
    for k in range(1, len(PRref)):  # первое значение не зашумляется, оно достоверное
        alpha = PRref[k] * 5 / 100
        PRref_k = norm_sample(mu=PRref[k], sigma=alpha * k)
        extra = 0  # можно увеличить колокол, а то граничные значения имеют обрезанные колоколы
        PRref_noised[k] = min(max(PRref_k, min_production + extra), max_production)

    # print("PRref_noised")
    # print(PRref_noised)
    # print()

    return [PRref, PRref_noised]

def init_Plant(num_of_elecs, init_states, delta_t):

    Plant = []
    IDs = []

    for j in range(num_of_elecs):
        elec = Electrolyser(j, delta_t)  # параметры по умолчанию
        [state, envTemperature, total_cost_of_work, total_run_out] = init_states[j]
        elec.set_init_state([state, envTemperature, total_cost_of_work, total_run_out])
        Plant.append(elec)
        IDs.append(j)

    return Plant

def formate_state(Electrolysers, desired_curve):
    # desired_curve это кривая желаемого значения выхода (в долях от максимальной выработки со всех электролизеров, т.е. от 500%)
    # со всех электролизеров на ближайшие N временных шагов. ширина шага равна Delta_t = 15 минут

    # print('desired_curve')
    # print(desired_curve)
    # print()

    assert max(desired_curve) <= 1

    # 6 часов с разрешением 15 минут => 24 значения в массиве
    x_desired_curve = desired_curve.copy()

    El_Targets = []
    El_OutputRate = []
    El_OutputRate_dot = []
    El_Temperatures = []
    El_States = []
    El_RunOuts = []

    for j in range(len(Electrolysers)):
        elec = Electrolysers[j]

        El_Targets.append(elec.getCurrentTarget())
        El_OutputRate.append(elec.getDinamics()[0])
        El_OutputRate_dot.append(elec.getDinamics()[1])
        El_Temperatures.append(elec.getTemperatureDinamics()[0])

        El_States.append(elec.states_list.index(elec.getState()))

        El_RunOuts.append(elec.getRunOut())

    x_El_Targets = np.array(El_Targets)
    x_El_OutputRate = np.array(El_OutputRate)
    x_El_OutputRate_dot = np.array(El_OutputRate_dot)

    max_value_El_Temperatures = Electrolysers[0].maxTemperature + 0.2
    x_El_Temperatures = np.array(El_Temperatures) / max_value_El_Temperatures

    x_El_States = np.array(El_States)

    x_El_States_one_hot = np.zeros((x_El_States.size, len(Electrolysers[0].states_list)))
    x_El_States_one_hot[np.arange(x_El_States.size), x_El_States] = 1

    x_El_States_one_hot = np.reshape(x_El_States_one_hot, (1, len(Electrolysers[0].states_list) * len(Electrolysers)))[0]

    x_El_RunOuts = np.array(El_RunOuts)

    El_RunOuts_norm_value = sum(x_El_RunOuts)

    if El_RunOuts_norm_value != 0:
        x_El_RunOuts /= El_RunOuts_norm_value # нормирую чтобы подать в сеть, но будут учитываться не абсалютные значения

    state = np.concatenate((x_desired_curve, x_El_Targets, x_El_OutputRate, x_El_OutputRate_dot, x_El_Temperatures,
                            x_El_States_one_hot, x_El_RunOuts))

    return state

def asymetric_plus(x, a, b):

    return (np.log(1 + np.e**(x)) * a)**2 + (np.log(1 + np.e**(-x)) / b)**2



