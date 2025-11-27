#!/usr/bin/env python
import os
from typing import Any
from pathlib import Path
from datetime import datetime
import argparse
import shlex
import shutil
import yaml
from htcondor.htcondor import Submit
from htcondor.htcondor import Schedd
from coolname import generate_slug


PROJECT_NAME = 'deepmuonreco'

def make_exp_name(config_file: Path, **kwargs) -> str:
    if exp_name := kwargs.get('exp', None):
        if debug_name := kwargs.get('debug', None):
            exp_name = debug_name
    else:
        with open(config_file, 'r') as file:
            base_config = yaml.safe_load(file)
            for each in base_config['defaults']:
                if 'exp' in each:
                    sub_config_name = each['exp']
                    break
            else:
                raise ValueError(f'exp not found in {config_file}')

        with open(config_file.parent / 'exp' / f'{sub_config_name}.yaml', 'r') as file:
            exp_config = yaml.safe_load(file)
        exp_name = exp_config['name']
    return exp_name


def make_run_name(run_name: str | None) -> str:
    run_name = run_name or generate_slug(pattern=2)
    now = datetime.now().strftime('%y%m%d-%H%M%S')
    return f'{now}_{run_name}'


def run(
    config_name: str,
    extra_args: str | None,
    memory: str,
    gpus: int,
    cpus: int,
    node_list: list[str],
    run_name: str | None = None,
    run_description: str | None = None,
    **kwargs,
) -> None:
    """
    Args:
    """

    if root_dir := os.getenv('PROJECT_ROOT'):
        root_dir = Path(root_dir).resolve()
    else:
        raise EnvironmentError('PROJECT_ROOT environment variable is not set')

    script_file_path = root_dir / 'train.py'
    if not script_file_path.exists():
        raise FileNotFoundError(f'Executable not found: {script_file_path}')
    print(f'{script_file_path=}')

    config_dir = root_dir / 'config'
    if not config_dir.exists():
        raise FileNotFoundError(f'Config directory not found: {config_dir}')
    print(f'{config_dir=}')

    config_file = config_dir / config_name
    config_file = config_file.with_suffix('.yaml')
    if not config_file.exists():
        raise FileNotFoundError(f'Config file not found: {config_file}')

    if executable_file_path := shutil.which('python'):
        executable_file_path = Path(executable_file_path)
    else:
        raise FileNotFoundError('Python executable not found in PATH')
    print(f'{executable_file_path=}')

    exp_name = make_exp_name(config_file, **kwargs)
    run_name = make_run_name(run_name)

    # NOTE: arguments
    arg_list = [
        str(script_file_path),
        '-cd', config_dir,
        '-cn', config_name,
    ]

    for key, value in kwargs.items():
        if value is not None:
            arg_list.append(f'{key}={value}')

    arg_list += [
        f'exp.name={exp_name}',
        f'run.name={run_name}',
        f'trainer.devices={gpus}',
        f'datamodule.num_workers={cpus - 1}', # FIXME:
    ]

    if extra_args is not None:
        arg_list += shlex.split(extra_args)

    arguments = ' '.join(map(str, arg_list))
    print(f'{arguments=}')

    # NOTE: condor logs
    condor_log_dir = root_dir / 'logs' / exp_name / run_name
    condor_log_dir.mkdir(parents=True, exist_ok=True)
    condor_log_path = condor_log_dir / 'condor'

    if run_description is not None:
        description_file_path = condor_log_dir / 'description.txt'
        with open(description_file_path, 'w') as stream:
            stream.write(run_description)

    job_batch_name = f'{PROJECT_NAME}.{exp_name}'

    requirements = '||'.join([f'(machine=="{each}.sscc.uos")' for each in node_list])
    print(f'{requirements=}')

    raw_submit: dict[str, Any] = {
        'universe': 'vanilla',
        'getenv': 'True',
        # job
        'executable': executable_file_path,
        'arguments': arguments,
        #
        'should_transfer_files': 'NO',
        # resources
        'request_cpus': cpus,
        'request_GPUs': gpus,
        'request_memory': memory,
        'max_retries': 0,
        'requirements': requirements, # which nodes to use
        #
        'JobBatchName': job_batch_name,
        'output': condor_log_path.with_suffix('.out'),
        'error': condor_log_path.with_suffix('.err'),
        'log': condor_log_path.with_suffix('.log'),
        'priority': 1000,
    }

    submit: dict[str, str] = {
        key: str(value)
        for key, value in raw_submit.items()
    }

    submit = Submit(submit)
    schedd = Schedd()
    schedd.submit(submit)

    print(f'🚀🚀🚀 submit {job_batch_name}')



def main():
    if root_dir := os.getenv('PROJECT_ROOT'):
        root_dir = Path(root_dir).resolve()
    else:
        raise EnvironmentError('PROJECT_ROOT environment variable is not set')

    config_dir = root_dir / 'config'
    if not config_dir.exists():
        raise FileNotFoundError(f'Config directory not found: {config_dir}')

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('-cn', '--config-name', type=str, default='tracker_track_selection', help='config name') # TODO: choices

    for sub_config_dir in config_dir.glob('*'):
        if not sub_config_dir.is_dir():
            continue
        name_flag = sub_config_dir.name.replace('_', '-')
        name_help = sub_config_dir.name.replace('_', ' ')
        parser.add_argument(f'--{name_flag}', type=str, help=f'{name_help} config dir')

    parser.add_argument('-a', '--extra-args', type=str, required=False, help='a list of yaml config files')
    parser.add_argument('-m', '--memory', type=str, default='64GB', help='memory')
    # TODO: when gpus > 1, update config_file
    parser.add_argument('--gpus', type=int, default=1, help='the number of GPUs')
    parser.add_argument('--cpus', type=int, default=3, help='the number of CPUs')
    parser.add_argument('-n', '--run-name', type=str, help='name of the run')
    parser.add_argument('-d', '--run-description', type=str, help='description of the run')
    parser.add_argument('--node', dest='node_list', type=str, nargs='+', default=['gpu01', 'gpu02', 'gpu03'])
    args = parser.parse_args()

    run(**vars(args))


if __name__ == "__main__":
    main()
