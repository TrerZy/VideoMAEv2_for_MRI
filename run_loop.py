import subprocess

from datetime import datetime
current_date = datetime.today().date()

log_out = ''
num_log = 1

gpu_rank = [0, 1, 3, 4]
lr = [1e-1, 1e-2, 1e-3, 1e-4]
master_port = [12300, 12310, 12320, 12330]

ratio = 4

for layer_decay in [0.3, 0.7]: 

    for drop_path in [0.5, 0.2]: 

        processes = []

        for i in [0, 1, 2, 3]:

            num_log_str = str(num_log).zfill(2)
            log_out = f'{current_date}_{num_log_str}_lr={lr[i]}_dp={drop_path}_ld={layer_decay}_{ratio}:1'
            num_log += 1

            process = subprocess.Popen(['bash', 'code/VideoMAEv2/run.sh', 
                log_out, str(ratio), str(gpu_rank[i]), str(lr[i]), str(drop_path), str(layer_decay), str(master_port[i])])
            
            processes.append(process)

        for process in processes:
            process.wait()