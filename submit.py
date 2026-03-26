#!/usr/bin/env python
from pathlib import Path
from datetime import datetime
import argparse
import shlex
import shutil
import yaml
from htcondor2 import Submit
from htcondor2 import Schedd
from coolname.impl import generate_slug


PROJECT_NAME = "muonly"


def make_exp_name(config_file: Path, **kwargs) -> str:
    if kwargs.get("debug"):
        return "debug"

    if exp_name_cli := kwargs.get("exp"):
        return exp_name_cli

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    if exp_name_config := config.get("exp", {}).get("name"):
        return exp_name_config

    defaults = config.get("defaults", [])

    default_exp = next((item["exp"] for item in defaults if "exp" in item), None)

    if default_exp:
        default_config_file = config_file.parent / "exp" / f"{default_exp}.yaml"
        with open(default_config_file, "r") as file:
            default_config = yaml.safe_load(file)
            exp_name_default_config = default_config.get("name")
        return exp_name_default_config

    raise ValueError(f"Experiment name not found in {config_file}")


def make_run_name(run_name: str | None) -> str:
    run_name = run_name or generate_slug(pattern=2)
    now = datetime.now().strftime("%y%m%d-%H%M%S")
    return f"{now}_{run_name}"


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
    if gpus < 1:
        raise ValueError("At least one GPU is required")
    if cpus < 1:
        raise ValueError("At least one CPU is required")


    root_dir = Path(__file__).parent.resolve()

    script_file_path = root_dir / "train.py"
    if not script_file_path.exists():
        raise FileNotFoundError(f"Script file not found: {script_file_path}")

    config_dir = root_dir / "config"
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    config_file = (config_dir / config_name).with_suffix(".yaml")
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    if executable := shutil.which("python"):
        executable = Path(executable)
    else:
        raise FileNotFoundError("Python executable not found in PATH")
    print(f"{executable=}")

    exp_name = make_exp_name(config_file, **kwargs)
    run_name = make_run_name(run_name)

    # NOTE: arguments
    arg_list = [
        str(script_file_path),
        "-cd",
        config_dir,
        "-cn",
        config_name,
    ]

    for key, value in kwargs.items():
        if value is not None:
            arg_list.append(f"{key}={value}")

    arg_list += [
        f"exp.name={exp_name}",
        f"run.name={run_name}",
        f"trainer.devices={gpus}",
        f"datamodule.num_workers={cpus - 1}",  # FIXME:
    ]

    if extra_args is not None:
        arg_list += shlex.split(extra_args)

    arguments = " ".join(map(str, arg_list))
    print(f"{arguments=}")

    # NOTE: condor logs
    condor_log_dir = root_dir / "logs" / exp_name / run_name
    condor_log_dir.mkdir(parents=True, exist_ok=True)
    condor_log_path = condor_log_dir / "condor"

    if run_description is not None:
        description_file_path = condor_log_dir / "description.txt"
        with open(description_file_path, "w") as stream:
            stream.write(run_description)

    job_batch_name = f"{PROJECT_NAME}.{exp_name}"

    requirements = "||".join([f'(machine=="{each}.sscc.uos")' for each in node_list])
    print(f"{requirements=}")

    submit_input = {
        "universe": "vanilla",
        "getenv": "True",
        # job
        "executable": executable,
        "arguments": arguments,
        #
        "should_transfer_files": "NO",
        # resources
        "request_cpus": cpus,
        "request_GPUs": gpus,
        "request_memory": memory,
        "max_retries": 0,
        "requirements": requirements,  # which nodes to use
        #
        "JobBatchName": job_batch_name,
        "output": condor_log_path.with_suffix(".out"),
        "error": condor_log_path.with_suffix(".err"),
        "log": condor_log_path.with_suffix(".log"),
        "priority": 1000,
    }

    submit_input = {key: str(value) for key, value in submit_input.items()}

    submit = Submit(submit_input)
    schedd = Schedd()
    submit_result = schedd.submit(submit)

    print(f"🚀🚀🚀 submit {job_batch_name} with cluster id {submit_result.cluster()}")


def main():
    root_dir = Path(__file__).parent.resolve()

    config_dir = root_dir / "config"
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-cn",
        "--config-name",
        type=str,
        default="tts",
        help="config name",
    )  # TODO: choices
    for sub_config_dir in config_dir.glob("*"):
        if not sub_config_dir.is_dir():
            continue
        name_flag = sub_config_dir.name.replace("_", "-")
        name_help = sub_config_dir.name.replace("_", " ")
        parser.add_argument(f"--{name_flag}", type=str, help=f"{name_help} config dir")

    parser.add_argument(
        "-a",
        "--extra-args",
        type=str,
        required=False,
        help="a list of yaml config files",
    )
    parser.add_argument("-m", "--memory", type=str, default="80GB", help="memory")
    # TODO: when gpus > 1, update config_file
    parser.add_argument("--gpus", type=int, default=1, help="the number of GPUs")
    parser.add_argument("--cpus", type=int, default=3, help="the number of CPUs")
    parser.add_argument("-n", "--run-name", type=str, help="name of the run")
    parser.add_argument(
        "-d", "--run-description", type=str, help="description of the run"
    )
    parser.add_argument(
        "--node",
        dest="node_list",
        type=str,
        nargs="+",
        default=["gpu01", "gpu02", "gpu03"],
    )
    args = parser.parse_args()

    run(**vars(args))


if __name__ == "__main__":
    main()
