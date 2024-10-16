import scripts.Simulation_config
import scripts.ModelTrainer as ModelTrainer

def run_smoketest():
    configs = scripts.Simulation_config.configs
    ModelTrainer().run(configs)
    print("Smoke test passed!")
    
if __name__ == "__main__":
    run_smoketest()
