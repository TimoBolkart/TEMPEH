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
import json
import argparse

# -----------------------------------------------------------------------------

def json_dict(string):
    try:
        return json.loads(string)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON string: {string}") from e

class BaseTrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)  
        self.hierachy = {}
        self.default_parameters = {}

    def parse(self, config_filename=''):
        self.initialize()

        # 1. load default parameters
        _, _ = self.parser.parse_known_args()
        args = self.parser.parse_args()
        self.opt = args

        cmd_args = {}
        for key in self.default_parameters:
            if self.cmd2name(key) in args:
                # If key is in command line arguments, use the command line argument
                cmd_args[self.cmd2name(key)] = getattr(self.opt, self.cmd2name(key))
            else:
                # If key is not in command line arguments, use the default argument
                setattr(self.opt, self.cmd2name(key), self.default_parameters[key])

        # 2. update parameters with parameters loaded from the config
        # possibly override options if config_filename is provided
        if config_filename != '':
            self.load_from_json(config_filename)
        elif getattr(self.opt, 'config_filename') != '':
            config_fname = self.opt.config_filename
            self.load_from_json(config_fname)
            # overwrite config path loaded from the file
            self.opt.config_filename = config_fname 

        # 3. update with parameters specified as command line arguments
        for key in cmd_args:
            setattr(self.opt, key, cmd_args[key])

        self.print_options()
        return self.opt

    @staticmethod
    def cmd2name(cmd):
        return cmd.replace('-', '_')

    @staticmethod
    def name2cmd(name):
        return name.replace('_', '-')

    def add_arg(self, cate, abbr, name, type, default):
        self.parser.add_argument('-'+abbr, '--'+self.name2cmd(name), type=type, default=default)
        if cate not in self.hierachy.keys():
            self.hierachy[cate] = []
        self.hierachy[cate].append( name )

    def initialize(self):
        # base
        self.add_arg( cate='base', abbr='s',   name='seed', type=str, default=0 )
        self.add_arg( cate='base', abbr='g',   name='gpu', type=str, default=0 )

        # train
        self.add_arg( cate='train', abbr='b',   name='batch-size', type=int, default=2)
        self.add_arg( cate='train', abbr='lr',  name='learning-rate', type=float, default=1e-3) # 1e-3
        self.add_arg( cate='train', abbr='lrt', name='lr-type', type=str, default='constant')   
        # self.add_arg( cate='train', abbr='lrt', name='lr-type', type=str, default='step')  
        self.add_arg( cate='train', abbr='gmn', name='gradient-max-norm', type=float, default=-1.0)   
        self.add_arg( cate='train', abbr='r',   name='resume-path', type=str, default='')

        # data
        # self.add_arg( cate='data', abbr='data-dir', name='dataset-directory', type=str, default='/ps/data/Faces/FaMoS')
        # self.add_arg( cate='data', abbr='reg-dir', name='processed-directory', type=str, default='/ps/project/facealignments/FaMoS/neutral_align_2023')
        # self.add_arg( cate='data', abbr='data-dir', name='dataset-directory', type=str, default='')
        # self.add_arg( cate='data', abbr='reg-dir', name='processed-directory', type=str, default='')

        # self.add_arg( cate='data', abbr='tdl', name='train-data-list-fname', type=str, default='/ps/project/famos/FaMoS/ToFu/seventy_subj__all_seq_frames_per_seq_40_head_rot_120_train.json')
        # self.add_arg( cate='data', abbr='vdl', name='val-data-list-fname', type=str, default='/ps/project/famos/FaMoS/ToFu/eight_subj__all_seq_frames_per_seq_5_val.json')
        # self.add_arg( cate='data', abbr='tdl', name='train-data-list-fname', type=str, default='/ps/project/famos/FaMoS/TEMPEH_data_to_publish/training_data/data_lists/seventy_subj__all_seq_frames_per_seq_40_head_rot_120_train.json')
        # self.add_arg( cate='data', abbr='vdl', name='val-data-list-fname', type=str, default='/ps/project/famos/FaMoS/TEMPEH_data_to_publish/training_data/data_lists/FaMoS_eigth_val_subjects.json')


        self.add_arg( cate='data', abbr='thread', name='thread-num', type=str, default=16)

        # register extra options
        self.initialize_extra()

        self.initialized = True

    def initialize_extra(self):
        # to be defined
        pass

    def print_options(self):
        message = ''
        message += '----------------- Options ---------------\n'
        categories = self.hierachy.keys()
        for cate in categories:
            message += '\n[{:}]:\n'.format(cate)
            for k in self.hierachy[cate]:
                v = getattr( self.opt, self.cmd2name(k) )
                comment = ''
                default = self.parser.get_default(self.cmd2name(k))
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        self.message = message

    def save_json(self, save_path):
        data = {}
        for cate in self.hierachy.keys():
            data[cate] = {}
            for k in self.hierachy[cate]:
                data[cate][self.cmd2name(k)] = getattr( self.opt, self.cmd2name(k) )
        import json
        with open(save_path, 'w') as fp:
            json.dump(data, fp, indent=4)
        print( "saved options to json file: %s" % (save_path) )

    def load_from_json(self, json_path):
        import json
        with open(json_path) as fp:
            data = json.load(fp)
        for cate in data.keys():
            for k in data[cate].keys():
                # import pdb; pdb.set_trace()
                setattr( self.opt, self.cmd2name(k), data[cate][self.cmd2name(k)] )
        print( "Options overwritten by json file: %s" % (json_path) )
