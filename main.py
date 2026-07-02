import hydra
from omegaconf import DictConfig

@hydra.main(config_path = 'config', config_name = 'config')
def main(config: DictConfig):
    pass

if __name__ == '__main__':
    main()
