import os
from pathlib import Path

env = 'julearn'

cwd = os.getcwd()

log_dir = Path(cwd) / 'logs' / 'run_models'
log_dir.mkdir(exist_ok=True, parents=True)


exec_string = '$(script) --features $(features) --cv kfold'

preamble = f"""
# The environment
universe       = vanilla
getenv         = True

# Resources
request_cpus   = 1
request_memory = 8G
request_disk   = 0

# Executable
initial_dir    = {cwd}
executable     = {cwd}/run_in_venv.sh
transfer_executable = False

arguments      = {env} python {exec_string}

# Logs
log            = {log_dir.as_posix()}/$(log_fname).log
output         = {log_dir.as_posix()}/$(log_fname).out
error          = {log_dir.as_posix()}/$(log_fname).err

"""

to_run = {
    '1_unimodal.py': [1, 2, 3, 4, 5, 6, 7],
    '2_unimodal_merged.py': [1, 2, 3, 4, 5, 6, 7],
    '3_multimodal.py': [1, 2, 3, 4, 5, 6, 7],
}
submit_fname = 'run_models.submit'

with open(submit_fname, 'w') as submit_file:
    submit_file.write(preamble)
    for t_script, t_features in to_run.items():
        for t_f in t_features:
            submit_file.write(f'script={t_script}\n')
            submit_file.write(
                f'log_fname=run_models{t_script}_{t_f}\n')
            submit_file.write(f'features={t_f}\n')
            submit_file.write('queue\n\n')
