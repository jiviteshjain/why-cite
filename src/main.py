import os

import hydra

import runners


@hydra.main(config_path='configs/', config_name='config')
def main(config):
    os.chdir(hydra.utils.get_original_cwd())

    runner = getattr(runners, config.training.file)

    if config.action == 'train':
        runner.train(config)
    elif config.action == 'test':
        runner.test(config)


if __name__ == '__main__':
    main()
