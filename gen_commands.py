# -*- coding: utf-8 -*-

import argparse
import os
import logging
import sys
import time
import platform
import pdb
import numpy


value_names = {
    1: "e0",
    0: "null",
    0.1: "e1",
    0.5: "5e1",
    0.01: "e2",
    0.03: "3e2",
    0.05: "5e2",
    0.001: "e3",
    0.0001: "e4",
    0.00001: "e5",
}
parameters = (
    ("lambda1", (1, 0.1, 0.01), False),
    ("lambda2", (1, 0.1, 0.01), False),
)


def process_parameter_command(index, tag, command, parameter_commands):
    assert index < len(parameters)

    arr = parameters[index]
    for value in arr[1]:    
        if value in value_names:
            native_tag = tag + "_%s%s" % (arr[0], value_names[value])
        else:
            native_tag = tag + "_%s%s" % (arr[0], str(value))
        native_command = command + " --%s%s %r" % ("lambda" if arr[2] else "", arr[0], value)
        if index == len(parameters) - 1:
            parameter_commands.append([native_tag, native_command])
            continue
        else:
            process_parameter_command(index + 1, native_tag, native_command, parameter_commands)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', type=str, default=r'/home/chenxu/datasets/pelvic/h5_data', help='path of the dataset')
    parser.add_argument('--checkpoint_dir', type=str, default=r'/home/chenxu/training/checkpoints/odada/pelvic', help="checkpoint file dir")
    parser.add_argument('--logfile_dir', type=str, default='outputs/pelvic', help="logfile dir")
    parser.add_argument('--output_file', type=str, default='commands.txt', help="checkpoint file dir")

    args = parser.parse_args()

    tag = "test"
    parameter_commands = []
    process_parameter_command(0, tag, "", parameter_commands)

    f = open(args.output_file, "w")
    for arr in parameter_commands:
        command = ("nohup python main_native.py --gpu %s --root_path %s --ckpt %s %s --epochs 100 > %s \n") % \
                  ("%d", args.data_dir, os.path.join(args.checkpoint_dir, arr[0]), arr[1],
                   os.path.join(args.logfile_dir, "%s.txt" % arr[0]))
        f.write(command)
