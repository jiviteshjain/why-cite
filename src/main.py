import os

import hydra

from runner import train, test


@hydra.main(config_path='configs/', config_name='acl_arc_test')
def main(config):
    os.chdir(hydra.utils.get_original_cwd())

    if config.action == 'train':
        train(config)
    elif config.action == 'test':
        test(config)


if __name__ == '__main__':
    main()