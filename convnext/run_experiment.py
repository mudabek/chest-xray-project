

# Create the evaluator
evaluator = nni.retiarii.evaluator.FunctionalEvaluator(evaluate_model)

exp = RetiariiExperiment(base_model, evaluator, [], search_strategy)
exp_config = RetiariiExeConfig('local')
exp_config.experiment_name = 'mnist_search'
exp_config.trial_concurrency = 2
exp_config.max_trial_number = 20
exp_config.training_service.use_active_gpu = False
exp.run(exp_config, 8081)


for model_code in exp.export_top_models(formatter='dict'):
    print(model_code)