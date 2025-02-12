import argparse
import gin
import os
from agents import util
import numpy as np
import multiprocessing as mp
import cma

class PIAgentTrain():
    def __init__(self, 
                 config, 
                 log_dir, 
                 load_model,
                 population_size=[16],
                 num_workers=-1, 
                 num_gpus=0,
                 max_iter=1,
                 save_interval=1,
                 seed=25,
                 reps=16,
                 init_sigma=0.01):
        super(PIAgentTrain, self).__init__()
        self.config=config
        self.log_dir=log_dir
        self.load_model = None
        self.population_size=population_size
        self.num_workers=num_workers
        self.num_gpus=num_gpus
        self.max_iter=max_iter
        self.save_interval=save_interval
        self.seed=seed
        self.reps=reps
        self.init_sigma=init_sigma
    solution = None
    task = None


    def worker_init(self, config_file, device_type, num_devices):
        #print("worker init")
        global task, solution
        gin.parse_config_file(config_file)
        task = util.create_task(logger=None)
        worker_id = int(mp.current_process().name.split('-')[-1])
        device = '{}:{}'.format(device_type, (worker_id - 1) % num_devices)
        solution = util.create_solution(device=device)


    def get_fitness(self, params):
        global task, solution
        params, task_seed, num_rollouts = params
        task.seed(task_seed)
        solution.set_params(params)
        scores = []
        for _ in range(num_rollouts):
            print("scores:",task.rollout(solution=solution, evaluation=False))
            scores.append(task.rollout(solution=solution, evaluation=False))
        return np.mean(scores)
    

    def save_params(self, solver, solution, model_path):
        solution.set_params(solver.result.xfavorite)
        solution.save(model_path)


    def learn(self):
        
        if self.num_workers < 0:
            self.num_workers = mp.cpu_count()
        gin.parse_config_file(self.config)
        if self.num_gpus <= 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

        self.num_workers = 1    
        #train_obs_single_LT7
        logger = util.create_logger(name='train_obs_single_LT12', log_dir=self.log_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        util.save_config(self.log_dir, self.config)
        logger.info('Logs and models will be saved in {}.'.format(self.log_dir))

        rnd = np.random.RandomState(seed=self.seed)
        solution = util.create_solution(device='cpu:0')
        num_params = solution.get_num_params()
        
        for pop_num in self.population_size:
            print("***population size number:", pop_num)
            if self.load_model is not None:
                solution.load(self.load_model)
                print('Loaded model from {}'.format(self.config.load_model))
                init_params = solution.get_params()
            else:
                init_params = None
            bestever = cma.optimization_tools.BestSolution()
            solver = cma.CMAEvolutionStrategy(
                x0=np.zeros(num_params) if init_params is None else init_params,
                sigma0=self.init_sigma,
                inopts={
                    'popsize': pop_num, #self.population_size,
                    'verb_append': bestever.evalsall,
                    'seed': self.seed if self.seed > 0 else 42,
                    'randn': np.random.randn,
                },
            )

            best_so_far = -float('Inf')
            ii32 = np.iinfo(np.int32)
            repeats = [self.reps] * pop_num #self.population_size
            
            device_type = 'cpu' if self.num_gpus <= 0 else 'cuda'
            num_devices = mp.cpu_count() if self.num_gpus <= 0 else self.num_gpus
            with mp.get_context('spawn').Pool(
                    initializer=self.worker_init,
                    initargs=(self.config, device_type, num_devices),
                    processes=self.num_workers,
            ) as pool:
                for n_iter in range(self.max_iter):
                    params_set = solver.ask() #returns a list of N-dimensional candidate solutions to be evaluated
                    task_seeds = [rnd.randint(0, ii32.max)] * pop_num #self.population_size
                    fitnesses = []
                    ss = 0
                    print("num of workers:",self.num_workers)
                    while ss < pop_num: #self.population_size:
                        #ee = ss + min(self.num_workers, self.population_size - ss)
                        print("ss:", ss)
                        ee = ss + min(self.num_workers, pop_num - ss)
                        fitnesses.append(
                            pool.map(func=self.get_fitness,
                                    iterable=zip(params_set[ss:ee],
                                                task_seeds[ss:ee],
                                                repeats[ss:ee]))
                        )
                        ss = ee
                    print("fitness append")
                    fitnesses = np.concatenate(fitnesses)
                    if isinstance(solver, cma.CMAEvolutionStrategy):
                        # CMA minimizes.
                        solver.tell(params_set, -fitnesses)
                        #solver.manage_plateaus(sigma_fac = 1.8)
                    else:
                        solver.tell(fitnesses)
                    bestever.update(solver.best)
                    logger.info(
                        'Iter={0}, '
                        'max={1:.2f}, avg={2:.2f}, min={3:.2f}, std={4:.2f}'.format(
                            n_iter, np.max(fitnesses), np.mean(fitnesses),
                            np.min(fitnesses), np.std(fitnesses)))

                    best_fitness = max(fitnesses)
                    if best_fitness > best_so_far:
                        best_so_far = best_fitness
                        model_path = os.path.join(self.log_dir, 'best.npz')
                        self.save_params(
                            solver=solver, solution=solution, model_path=model_path)
                        logger.info('Best model updated, score={}'.format(best_fitness))
                        
                    if (n_iter + 1) % self.save_interval == 0:
                        model_path = os.path.join(
                            self.log_dir, 'iter_{}.npz'.format(n_iter + 1))
                        self.save_params(
                            solver=solver, solution=solution, model_path=model_path)

#if __name__ == '__main__':
#    learn()
       
    



