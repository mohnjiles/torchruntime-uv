import os
import shutil
import platform
import threading
import subprocess

OS_NAME = platform.system()


def locate_conda():
    which = "where" if OS_NAME == "Windows" else "which"
    conda = subprocess.getoutput(f"{which} conda")
    conda = conda.split("\n")
    conda = conda[0].strip()

    return conda


CONDA = locate_conda()
print("conda: ", CONDA)


def create_environment(env_dir, torch_platform, python_version="3.10"):
    env_dir = os.path.abspath(env_dir)
    print(f"Creating environment with {torch_platform} in {env_dir}..")

    # create the conda env
    run([CONDA, "create", "-y", "--prefix", env_dir])

    print("Installing packages..")

    # install python and git in the conda env
    run([CONDA, "install", "-y", "--prefix", env_dir, "-c", "conda-forge", f"python={python_version}", "git"])

    # print info
    run_in_conda(env_dir, ["git", "--version"])
    run_in_conda(env_dir, ["python", "--version"])

    # install the appropriate version of torch using torchruntime
    run_in_conda(env_dir, ["python", "-m", "pip", "install", "torchruntime"])
    run_in_conda(
        env_dir, ["python", "-m", "torchruntime", "install", "torch", "torchvision", "--platform", torch_platform]
    )


def delete_environment(env_dir):
    shutil.rmtree(env_dir)


def get_env(env_dir):
    if not os.path.exists(env_dir):
        raise RuntimeError("The system folder is missing!")

    python_exe = f"bin/python" if OS_NAME != "Windows" else f"python.exe"

    python_version = subprocess.getoutput(f"{env_dir}/{python_exe} --version")
    python_version = python_version.strip().split(" ")[1]
    python_version = ".".join(python_version.split(".")[:2])

    env_entries = {
        "PATH": [
            f"{env_dir}",
            f"{env_dir}/bin",
            f"{env_dir}/Library/bin",
            f"{env_dir}/Scripts",
            f"{env_dir}/usr/bin",
        ],
        "PYTHONPATH": [
            f"{env_dir}",
            f"{env_dir}/lib/site-packages",
            f"{env_dir}/lib/python{python_version}/site-packages",
        ],
        "PYTHONHOME": [],
    }

    if OS_NAME == "Windows":
        env_entries["PATH"].append("C:/Windows/System32")
        env_entries["PATH"].append("C:/Windows/System32/wbem")
        env_entries["PYTHONNOUSERSITE"] = ["1"]
        env_entries["PYTHON"] = [f"{env_dir}/python"]
        env_entries["GIT"] = [f"{env_dir}/Library/bin/git"]
    else:
        env_entries["PATH"].append("/bin")
        env_entries["PATH"].append("/usr/bin")
        env_entries["PATH"].append("/usr/sbin")
        env_entries["PYTHONNOUSERSITE"] = ["y"]
        env_entries["PYTHON"] = [f"{env_dir}/bin/python"]
        env_entries["GIT"] = [f"{env_dir}/bin/git"]

    env = dict(os.environ)
    for key, paths in env_entries.items():
        paths = [p.replace("/", os.path.sep) for p in paths]
        paths = os.pathsep.join(paths)

        env[key] = paths

    return env


def read_output(pipe, prefix=""):
    while True:
        output = pipe.readline()
        if output:
            print(f"{prefix}{output.decode('utf-8')}", end="")
        else:
            break  # Pipe is closed, subprocess has likely exited


def run(cmds: list, cwd=None, env=None, stream_output=True, wait=True, output_prefix=""):
    p = subprocess.Popen(cmds, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if stream_output:
        output_thread = threading.Thread(target=read_output, args=(p.stdout, output_prefix))
        output_thread.start()

    if wait:
        p.wait()

    return p


def run_in_conda(env_dir, cmds: list, *args, **kwargs):
    env = get_env(env_dir)

    cmds = [CONDA, "run", "--no-capture-output", "--prefix", env_dir] + cmds
    return run(cmds, env=env, *args, **kwargs)
