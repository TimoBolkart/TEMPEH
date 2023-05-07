"""
Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
holder of all proprietary rights on this computer program.
Using this computer program means that you agree to the terms 
in the LICENSE file included with this software distribution. 
Any use not explicitly granted by the LICENSE is prohibited.

Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

For comments or questions, please email us at tempeh@tue.mpg.de
"""

import os
from pathlib import Path
from option_handler.train_options_local import TrainOptions


def get_date_string():
    from datetime import datetime
    mydate = datetime.now()
    return '%s%02d__%02d-%02d-%02d' % (mydate.strftime("%B"), mydate.day, mydate.hour, mydate.minute, mydate.second)

def execute_locally(config_fname):
    from trainer.local_trainer import run
    run(config_fname=config_fname)

def train():
    parser = TrainOptions()
    config_args = parser.parse()

    if os.path.exists(config_args.config_filename):
        config_fname = config_args.config_filename
    else:
        if config_args.experiment_id == '':
            config_args.experiment_id = 'refinement__' + get_date_string()
        else:
            config_args.experiment_id = 'refinement__' + config_args.experiment_id + '__' + get_date_string()

        output_directory = os.path.join(config_args.model_directory, config_args.experiment_id)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        config_fname = os.path.join(output_directory, 'config.json')
        parser.save_json(config_fname)    

    execute_locally(config_fname)

if __name__ == '__main__':
    train()
    print('Done')